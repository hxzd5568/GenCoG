import tvm
from tvm import relay
import collections
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import GlobalVar
from tvm.relay.analysis import free_vars, free_type_vars
from tvm.relay import create_executor, transform
from tvm.relay.transform import gradient
from tvm.relay.prelude import Prelude
from tvm.relay.testing import (
    make_nat_expr,
    run_infer_type,
    check_grad,
    rand,
    count_ops,
)
import tvm.relay.op as op


target = 'llvm'
# Create two simple Relay programs
@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

def opt_tvm_ir(mod):
    # 创建优化器
    opt_passes = tvm.transform.Sequential([
        relay.transform.CanonicalizeOps(), # 规范化操作，消除多余的操作
        relay.transform.FoldScaleAxis(), # 折叠缩放轴
        relay.transform.FoldConstant(), # 折叠常数
        relay.transform.CombineParallelConv2D(), # 将相邻的卷积合并为一个卷积
        relay.transform.FoldScaleAxis(), # 再次折叠缩放轴
        #relay.transform.FuseOps(), # 将相邻的操作融合为一个操作
        relay.transform.CanonicalizeCast(), # 规范化类型转换操作
        relay.transform.SimplifyInference(), # 进行简化推理
        relay.transform.CanonicalizeOps(), # 规范化操作，消除多余的操作
    ])
    # 运行优化器
    with tvm.transform.PassContext(opt_level=3):#instruments=[PrintIR()]
        mod = opt_passes(mod)
    return mod


shape = (10,)
dtype = "float32"
t = relay.TensorType(shape, dtype)
x1 = relay.var('x1', t)
y1 = relay.var('y1', t)
z1 = relay.add(x1, y1)
f1 = relay.Function([x1, y1], z1)

x2 = relay.var('x2', t)
y2 = relay.var('y2', t)
z2 = relay.add(x2, y2)
f2 = relay.Function([x2, y2], z2)
mod = tvm.IRModule()
mod['f1']=f1
mod5 = opt_tvm_ir(mod)
f5 =mod5['f1']

new_x = relay.var('new_x', t)
new_y = relay.var('new_y', t)

new_z1 = relay.Call(f5, [new_x, new_y])
new_z2 = relay.Call(f2, [new_x, new_y])
new_z = relay.add(new_z1, new_z2)
new_f = relay.Function([new_x, new_y], new_z)
new_f2 = run_infer_type(new_f)
print(new_f2)
'''
# with open('/home/zichaox/tvm/Fuzzfp/Gencog/tempconbr','r') as f:
#     mod = relay.fromtext(f)
# new_f2 = mod['main']
'''


back_func = run_infer_type(gradient(new_f2))
mod = tvm.IRModule()


ccompilef= create_executor().evaluate(back_func)



x =  tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
y =  tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
forward, (grad1,grad2) = create_executor(mod=mod).evaluate(back_func)(x,y)
tvm.testing.assert_allclose(grad1.numpy(), 4 * np.ones_like(x.numpy()))



# shape = (10,)
# dtype = "float32"
# t = relay.TensorType(shape, dtype)

# x1 = relay.var('x1', t)
# y1 = relay.var('y1', t)
# z1 = relay.add(x1, y1)
# f1 = relay.Function([x1, y1], z1)

# x2 = relay.var('x2', t)
# y2 = relay.var('y2', t)
# z2 = relay.add(x2, y2)
# f2 = relay.Function([x2, y2], z2)

# new_x = relay.var('new_x', t)
# new_y = relay.var('new_y', t)
# mod = tvm.IRModule()
# d1 = GlobalVar("f11")
# d2 = GlobalVar("f22")
# mod[d1]=f1
# mod5 = opt_tvm_ir(mod)
# f5 =mod5[d1]
# mod[d2]=f2

# # Create a new Relay program that subtracts the results of the two previous programs
# new_x = relay.var('new_x', t)
# new_y = relay.var('new_y', t)

# new_z1 = relay.Call(f2, [new_x, new_y])
# new_z2 = relay.Call(f5, [new_x, new_y])
# new_z = relay.add(new_z1, new_z2)
# new_f = relay.Function([new_x, new_y], new_z)
# new_f2 = run_infer_type(new_f)
# mod['main']=new_f2
# print(mod.astext())
# exit()
# g = GlobalVar("grad")
# back_func = run_infer_type(gradient(new_f2))
# mod[g]=back_func
# x =  tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
# y =  tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
# forward, (grad,) = create_executor(mod=mod).evaluate(back_func)(x,y)
#compiled_f(x,y)

# forward, (grad,) = create_executor().evaluate(back_func)(x,y)
# tvm.testing.assert_allclose(grad.numpy(), 4 * np.ones_like(x.numpy()))

# Compile the new function

# mod = tvm.IRModule.from_expr(new_f)
# executor = relay.create_executor(kind='debug',target = 'llvm')
# compiled_f = executor.evaluate(mod['main'])
# # Evaluate the new function with inputs
# import numpy as np
# input_data1 = tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
# input_data2 = tvm.nd.array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('float32'))
# output_data = compiled_f(input_data1, input_data2)
# print(output_data.asnumpy())

