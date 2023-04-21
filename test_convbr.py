import os
import numpy as np
import queue
import shutil
import os.path
import random
from enum import IntEnum, auto
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError
from sys import stdout
from time import strftime
from numpy.random import Generator, PCG64
from tqdm import tqdm
from gencog.expr.ty import  DataType, ValueType, BOOL, INT, FLOAT
from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry
from gencog.debug import ModuleError
from typing import Iterable, List, cast, Optional, Dict
from fuzzutils import tactics

TensorDict = Dict[str, np.ndarray]

'''
### tests
a = np.arange(128*3072)
a = np.reshape(a,[128,3072])
print(a)
for i in range(6):
    print(tactics.f1(a))
for i in range(6):
    print(tactics.f2(a))
for i in range(6):
    print(tactics.f3(a))
for i in range(6):
    print(tactics.f4(a))
for i in range(6):
    print(tactics.f5(a))

'''
args = Namespace()
imageshape = cast(List[int],[128,3072])
imagedtype = DataType.f(16)
imagefile = '/data/cifar-10image/cifar-10-batches-py/data_batch_1'
Lastfuzztimes =3
Maxfuzztrytimes = 5e2
usedseed =[None]*10
# !!! rundiff add 

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    args = p.parse_args()

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

def build_mod(mod: IRModule, opt_level: int, params: Optional[TensorDict] = None):
    with transform.PassContext(opt_level=opt_level,
                               disabled_pass=['AlterOpLayout', 'CombineParallelDense']):
        lib = relay.build(mod, target='llvm', params=params)
    return GraphModule(lib['default'](cpu()))
def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    gmod.run(**inputs)
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

def unpickleimage(file:str):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def forksubprocess(gen:GraphGenerator, rng:Generator, path:str, case_id:int, env:dict, mode:str)-> int:
    graph = gen.imagegenerate(imageshape=imageshape, imagedtype=imagedtype)
    code = print_relay(graph)
    # Write code to case directory
    case_path = path
    with open(os.path.join(case_path, 'code.txt'), 'w') as f:
        f.write(code)
    # Run subprocess
    cmd = ['python', '_fuzzrun_ps.py', f'-d={case_path}', '-e', f'-s={rng.integers(2 ** 63)}', f'-m={mode}']
    keep_dir = False
    try: #!! pack them
        rcode = run(cmd, env=env, check=True, timeout=60, stdout=open('tempreturn', 'w'))
        if(rcode.returncode ==0):
            if(imagedtype.__str__()=='float16'):
                diff = np.float16(open('tempreturn','r').read()\
                                  .strip('\n').strip('[').strip(']'))
            else:
                diff = np.float32(open('tempreturn','r').read()\
                                  .strip('\n').strip('[').strip(']'))
            print('initdiff: ',diff)
            return diff
    except CalledProcessError:
        print(f'Error detected in initial check of case {case_id}.')
        keep_dir = True
        return None
    except TimeoutExpired:
        print(f'Case {case_id} timed out.')
        return None
def rundiff(main_fn:relay.function.Function, gmod1:GraphModule,gmod5:GraphModule,input:np):
    input = np.clip(input,0,1)
    var = main_fn.params[0:1][0]
    var_ty = var.checked_type
    inputarr = {var.name_hint: np.reshape(input,[128,32,32,3]).astype(var_ty.dtype)}

    outs1 = run_gmod(gmod1,inputarr)
    outs5 = run_gmod(gmod5,inputarr)
    tempdiff = np.zeros(1)
    for i, (o, ro) in enumerate(zip(outs1, outs5)):
        tempdiff =max( np.linalg.norm(o/1000-ro/1000)/o.size,tempdiff)
    return tempdiff
def endhandle(progress, keep_dir,case_path):
        if not keep_dir:
                os.remove(os.path.join(case_path, 'code.txt'))
                shutil.rmtree(case_path)
        elif(len( os.listdir(case_path) )==3):
                os.remove(os.path.join(case_path, 'code.txt'))
                shutil.rmtree(case_path)
        progress.update()
def cross(main_fn:relay.function.Function,inputo:np,input2:np,gmod1:GraphModule,gmod5:GraphModule):
    input = inputo.copy()
    #get input
    origindiff = rundiff(main_fn,gmod1,gmod5,input)
    sindex = random.randint(32,64)
    inputimage = input2[random.randint(0,9)]
    input[sindex:][:]=inputimage[sindex:][:]
    # valide input
    diff = rundiff(main_fn,gmod1,gmod5,input)
    if diff>origindiff:
        return input
    else:
        return None

def main():
    with open('/home/zichaox/tvm/Fuzzfp/Gencog/tempconbr', 'r') as f:
            code = f.read()
            mod = relay.parse(code)
    main_fn = mod['main']
    # Generation loop
    progress = tqdm(file=stdout)
    imagedict = unpickleimage(imagefile)
    imagedataset = imagedict[b'data']/255.0 # print(len(imagedict[b'data'][0:128])) #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    path = os.path.join(args.output, strftime('convbrrun-%Y%m%d-%H%M%S'))


    os.mkdir(path)

    while True:
        # base information of process
        case_id = str(progress.n)
        case_path = os.path.join(path, case_id)
        if not os.path.exists(path):
            os.mkdir(path)
        os.mkdir(case_path)

        # Initial check for runablility
        rng = Generator(PCG64(seed=args.seed+random.randint(0,1e4)))

        # prepare runmodule

        params = gen_tensor_value_dict(main_fn.params[1:], rng)
        gmod1 = build_mod(mod, 1, params=params)
        gmod5 = build_mod(mod, 5, params=params)
        inputarr = np.random.normal(size=imageshape)
        prediff = rundiff(main_fn,gmod1,gmod5,inputarr)


        # inital imageseed
        diff = rundiff(main_fn,gmod1,gmod5,imagedataset[0:128])
        iter = 1
        reflag =0
        while( diff <= prediff ):
            if(iter==5 and diff<1e-7):
                print(f'optimization seems true : case {case_id}')
                reflag=1
                break
            if(iter>=5000/128-1):
                print(f'failed to find potensial fuzz input for : case {case_id}')
                reflag=1
                break
            inputarr = imagedataset[0+iter*128:128+iter*128]
            diff =max(diff,rundiff(main_fn,gmod1,gmod5,inputarr))
            iter+=1
        if(reflag==1):
            endhandle(progress, 0, case_path)
            continue

        q = queue.Queue()
        q.put(inputarr.copy())
        keep_dir = False
        # fuzz with 6 tactics
        lastdiff = diff
        lastinputarr = inputarr
        trys =0

        while not q.empty():
            
            trys += 1
            if trys> Maxfuzztrytimes:
                break
            imageseed = q.get()
            print(imageseed.ndim,imageseed.shape==(128,3072))
            # update usedseed
            if len(usedseed)==10:
                usedseed[random.randint(0,9)]=imageseed
            else:
                usedseed.append(imageseed)
            tac = tactics()
            #  fuzz loop
            funcl = [tac.f0,tac.f1,tac.f2,tac.f3]
            seedinitdiff = diff
            for mutateseed in funcl:
                lasttrytimes = Lastfuzztimes
                #enhance 
                inputarr = mutateseed(imageseed)

                while(lasttrytimes!=0):
                    diff = rundiff(main_fn,gmod1,gmod5,inputarr)
                    
                    if(diff>lastdiff):
                        lasttrytimes = Lastfuzztimes
                        lastdiff = diff
                        lastinputarr = inputarr
                    else:
                        lasttrytimes -= 1
                    # !!! realize update weight 

                    # enhance
                    inputarr = mutateseed(inputarr)
            
            if (lastdiff > seedinitdiff):
                q.put(lastinputarr)
                cseedinputimageseed = cross(main_fn,lastinputarr,usedseed,gmod1,gmod5)
                if(cseedinputimageseed!=None):
                    q.put(cseedinputimageseed)

        # dump norm output diff
        ninput = np.random.normal(0.5, 1,size = imageshape)
        ndiff = rundiff(main_fn,gmod1, gmod5, ninput)
        # dump curthy output diff
        cinput = np.random.standard_cauchy(size =imageshape)
        cdiff = rundiff(main_fn,gmod1, gmod5, cinput)
        # Prepare ab/rel error
        # dump submaxdiff

        msg = f'Computation diff in runmodule :\n' \
                          f'Fuzz:\n' \
                          f'{np.array_repr(lastdiff)}\n' \
                          f'Norm:\n' \
                          f'{np.array_repr(ndiff)}\n'\
                          f'Cauthy:\n' \
                          f'{np.array_repr(cdiff)}\n'
        merror = ModuleError('fuzz', mod.astext(), msg, opt_level=5,
                                      inputs=lastinputarr, params=params)
        merror.fuzzreport(case_path)
        #  if files include 3 files then delete it
        endhandle(progress, 1, case_path)
        exit()

if __name__ == '__main__':
    parse_args()
    main()
