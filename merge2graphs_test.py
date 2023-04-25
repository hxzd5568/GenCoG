import tvm 
target = tvm.target.Target("llvm", host="llvm")
import tvm
import numpy as np
from tvm import relay
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
args = Namespace()
imageshape = cast(List[int],[128,3072])
imagedtype = DataType.f(16)
imagefile = '/data/cifar-10image/cifar-10-batches-py/data_batch_1'
Lastfuzztimes =3
Maxfuzztrytimes = 5e2
usedseed =[None]*10

# 创建一个简单的Relay计算图
numoftest = 500
data_shape = (128, 32, 32, 3)
# kernel_shape = (32, 3, 3, 3)
dtype = "float32"

g1='#[version = "0.0.5"]\
def @main(%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {\
  %0 = nn.conv2d(%x, %weight1, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");\
  %1 = nn.relu(%0);\
  %2 = nn.conv2d(%1, %weight2, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");\
  nn.relu(%2)\
}\
fn(%1x: Tensor[(1, 56, 56, 64), float32], %1weight1: Tensor[(3, 3, 64, 32), float32], %1weight2: Tensor[(3, 3, 32, 32), float32]) {\
  %0 = nn.conv2d(%1x, %1weight1, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");\
  %1 = nn.relu(%0);\
  %2 = nn.conv2d(%1, %w1eight2, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");\
  nn.relu(%2)\
}'
def my_func1():
    x = relay.var('x', shape=(10,))
    y = relay.var('y', shape=(10,))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    return mod
def my_func2():
    x = relay.var('x', shape=(10,))
    y = relay.var('y', shape=(10,))
    z = relay.multiply(x, y)
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    return mod
def get_convbr(path:str):
    if path == '':
        with open('/home/zichaox/tvm/Fuzzfp/Gencog/tempconbr', 'r') as f:
            code = f.read()
            mod = relay.parse(code)
        return mod
    else:
        with open('/home/zichaox/tvm/tempconbr3', 'r') as f:
            code = f.read()
            mod = relay.parse(code)
        return mod

# 定义一个函数，用于打印优化后的TVM Tensor IR图
def opt_tvm_ir(mod):
    # 创建优化器
    opt_passes = tvm.transform.Sequential([
        relay.transform.CanonicalizeOps(), # 规范化操作，消除多余的操作
        relay.transform.FoldScaleAxis(), # 折叠缩放轴
        relay.transform.FoldConstant(), # 折叠常数
        relay.transform.CombineParallelConv2D(), # 将相邻的卷积合并为一个卷积
        relay.transform.FoldScaleAxis(), # 再次折叠缩放轴
        relay.transform.FuseOps(), # 将相邻的操作融合为一个操作
        relay.transform.CanonicalizeCast(), # 规范化类型转换操作
        relay.transform.SimplifyInference(), # 进行简化推理
    ])
    
    # 运行优化器
    with tvm.transform.PassContext(opt_level=3):
        mod = opt_passes(mod)
    return mod
def nonopt_tvm_ir(mod):
    # 创建优化器
    opt_passes = tvm.transform.Sequential([
    ])
    
    # 运行优化器
    with tvm.transform.PassContext(opt_level=1):
        mod = opt_passes(mod)
    return mod
# 运行示例程序

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
from tvm import relay
from tvm.relay import testing
if __name__ == "__main__":
    rng = Generator(PCG64(random.randint(0,1e4)))
    testing.gradient()
    with open('/home/zichaox/tvm/Fuzzfp/Gencog/tempconbr', 'r') as f:
        graph = f.read()
        mod = tvm.parser.fromtext(graph)

    # vars5 = relay.analysis.free_vars(mod5['main'].body)
    # vars1 = relay.analysis.free_vars(mod1['main'].body)
    # expr5 = relay.expr.Call(mod5["main"],vars5)
    # expr1 = relay.expr.Call(mod1["main"],vars1)
    # new_z = relay.subtract(expr1,expr5)
    # new_w = relay.multiply(new_z,new_z)
    # new_f = relay.Function(list(vars1)+list(vars5), new_w)
    # mod = tvm.IRModule.from_expr(new_f)
    main_fn = mod['main']
    params = gen_tensor_value_dict(main_fn.params[:], rng)
    with tvm.transform.PassContext(opt_level=1,
                                    ):#ForwardFoldScaleAxis,disabled_pass=['ForwardFoldScaleAxis','BackwardFoldScaleAxis']
                                    #required_pass=['FastMath'],
            graph1, lib1, params1 = relay.build(mod, target, params=params)
        # from tvm.contrib.debugger.debug_executor import GraphModuleDebug

        # paras = "./pass05"
        # m = GraphModuleDebug(
        #     lib["debug_create"]("default", dev),
        #     [dev],
        #     lib.graph_json,
        #     dump_root=paras,
        # )
    from tvm.contrib import graph_executor
    m1 = graph_executor.create(graph1, lib1, device=tvm.cpu(0))
    # !!! temp  numoftest 
    numoftest =1
    for i in range(numoftest):
        input_arr = np.random.uniform(0, 1, size=data_shape).astype(dtype)
        #y_arr = model.predict(input_arr)
        m1.set_input("x",tvm.nd.array(input_arr))
        m1.set_input(**params1)
        m1.run()
        #tvm_out1 = m1.get_output(0, tvm.nd.empty(((64, 112, 112, 32)), "float32")).numpy()
        tvm_out1 = m1.get_output(0).numpy()
        if(i%30==0):
             print(np.sqrt(np.mean(tvm_out1)))
    


