"""按稳定题号输出数学与系统练习的参考中间状态。"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
from .mathlab import attention, cosine
from .agent import execute, strict_json, run
from .retrieval import search
from .runtime import Store, assemble
from .harness import Queue, communication_cost
from .mcp_stdio import client_demo


def solve(code):
    if code.startswith('HA-'):
        from .harness_lab import exercise
        return exercise(code)
    if code.startswith('RT-'):
        from .persistence_lab import exercise
        return exercise(code)
    if code.startswith('PE-'):
        from .ecosystem_lab import exercise
        return exercise(code)
    if code.startswith('RA-'):
        from .rag_agent_lab import exercise
        return exercise(code)
    if code.startswith('MM-'):
        from .visual_audio import exercise
        return exercise(code)
    if code.startswith('PT-'):
        from .posttraining import exercise
        return exercise(code)
    # 原数字题号需要的训练模块在此加载，RT 等命名空间不预先导入 PyTorch。
    from .networks import temporal_gradients, sequence_gradient_norms, cache_experiment
    from .language import combination_experiment, multimodal
    from .language_details import cache_failure
    from .learning import experiment
    a, b = np.array([3.,4.]), np.array([4.,0.])
    w = np.array([[2.,1.],[-1.,1.]])
    if code == '01-01':
        # 同一分量排列为三种形状，只有二维转置交换轴。
        values = [a,a[:,None],a[None,:]]
        return [{'shape':x.shape,'transpose':x.T.shape} for x in values]
    if code == '01-02':
        return {'mapped':(w@b).tolist(),'batch':(np.stack([a,b])@w.T).tolist(),
                'elementwise':(w*np.stack([a,b])).tolist()}
    if code == '01-03':
        matrix = np.array([[1.,2.],[2.,4.]])
        solution = np.linalg.lstsq(matrix,[3.,7.],rcond=None)[0]
        return {'rank':int(np.linalg.matrix_rank(matrix)),'null':(matrix@[-2.,1.]).tolist(),
                'inconsistent_residual':(matrix@solution-[3.,7.]).tolist()}
    if code == '01-04':
        basis = np.array([[1.,1.],[1.,-1.]])
        coordinates = np.linalg.solve(basis,a)
        return {'coordinates':coordinates.tolist(),'restored':(basis@coordinates).tolist()}
    if code == '01-05':
        coefficient = a@b/(b@b)
        projection = coefficient*b
        return {'coefficient':coefficient,'projection':projection.tolist(),
                'residual':(a-projection).tolist(),'orthogonal':float((a-projection)@b)}
    if code == '01-06':
        scale = np.diag([1.,.1])
        # 输入坐标变化后，逆变换进入系数以保存预测。
        new_w = w@np.linalg.inv(scale)
        return {'scaled':(scale@a).tolist(),'old_cosine':cosine(a,b),
                'new_cosine':cosine(scale@a,scale@b),'new_w':new_w.tolist(),
                'preserved_output':(new_w@scale@a).tolist()}
    if code == '01-07':
        x = np.array([[1.,2.],[2.,4.2],[3.,6.]])
        centered = x-x.mean(0)
        u,s,vh = np.linalg.svd(centered,full_matrices=False)
        errors = [float(np.square(centered-(u[:,:k]*s[:k])@vh[:k]).sum()) for k in [1,2]]
        return {'singular':s.tolist(),'errors_1d_2d':errors}
    if code == '01-08':
        loss = lambda value:(3*value-4)**2/2
        eps = 1e-6
        return {'finite_difference':(loss(1+eps)-loss(1-eps))/(2*eps),
                'eta_01':{'w':1.3,'loss':loss(1.3)},'eta_1':{'w':4,'loss':loss(4)}}
    if code == '02-01':
        x, y = np.arange(1.,5.),np.array([3.,5.,7.,9.])
        dw,db = np.mean(-y*x),np.mean(-y)
        weights,bias = -.1*dw,-.1*db
        return {'dw':dw,'db':db,'w':weights,'b':bias,
                'predictions':(weights*x+bias).tolist(),
                'loss':float(np.mean((weights*x+bias-y)**2)/2)}
    if code == '02-02':
        p,y=.75,1
        upstream = -y/p+(1-y)/(1-p)
        return {'dL_dp':upstream,'dp_dz':p*(1-p),'dL_dz':upstream*p*(1-p)}
    if code == '02-03':
        return {'training_mean':float(np.mean([1.,2.])),
                'leaked_mean':float(np.mean([1.,2.,100.]))}
    if code == '02-04':
        result = experiment()
        return {k:result[k] for k in ['thresholds','threshold','tuned_logistic_test']}
    if code == '03-01':
        # 从旧图依次得到 dL/dy、dL/dh、dL/dz，再分到三份参数。
        old_h, old_v, target, x = 7., 2., 4., 3.
        d_prediction = old_v*old_h-target
        d_hidden = d_prediction*old_v
        d_linear = d_hidden  # 当前 z=7>0，ReLU 的局部导数为 1。
        gradients = {'w': d_linear*x, 'b': d_linear, 'v': d_prediction*old_h}
        # 三个参数同时使用旧梯度更新，之后重新前向。
        w1,b1,v1 = (2-.01*gradients['w'], 1-.01*gradients['b'],
                    2-.01*gradients['v'])
        prediction=v1*max(3*w1+b1,0)
        return {'old_gradients':gradients,'w':w1,'b':b1,'v':v1,
                'prediction':prediction,'loss':(prediction-4)**2/2}
    if code in ['03-02','03-03']:
        return temporal_gradients()
    if code == '03-04':
        scores,weights,output=attention([[0,1]],[[1,0],[0,1]],[[2,0],[0,4]])
        return {'scores':scores.tolist(),'weights':weights.tolist(),'output':output.tolist(),
                'rnn_gradients':sequence_gradient_norms()}
    if code == '04-01':
        text='苹果 AI'
        ids=list(text.encode())
        return {'characters':len(text),'tokens':len(ids),'ids':ids,
                'input_shape':[1,len(ids)],'embedding_shape':[1,len(ids),32]}
    if code == '04-02':
        x=np.array([[1.,0.],[0.,1.],[1.,1.]])
        scores,weights,output=attention(x,x,x,causal=True)
        # JSON 不支持数值负无穷，屏蔽分数用字符串表示；权重仍是正常浮点数。
        return {'scores':[[float(v) if np.isfinite(v) else '-inf' for v in row] for row in scores],
                'weights':weights.tolist(),'output':output.tolist()}
    if code == '04-03':
        return cache_failure()
    if code == '04-04':
        return multimodal()
    if code == '04-05':
        return combination_experiment()
    if code == '04-06':
        from .language_foundations import attention_backward
        return attention_backward()
    if code == '04-07':
        from .language_foundations import heads_and_positions
        return heads_and_positions()
    if code == '04-08':
        from .language_foundations import model_cache
        return model_cache()
    if code == '05-01':
        result=run('切片共享')
        return {'saved_roles':[m['role'] for m in result['messages']],
                'sent_roles':[[m['role'] for m in t['input']] for t in result['trace']],
                'trace':result['trace']}
    if code == '05-02':
        from .application_details import rank_example
        return rank_example()
    if code == '05-03':
        errors=[]
        for name,args in [('reading_minutes',{'pages':True,'questions':4}),
                          ('search_notes',{'query':'切片','owner':'bob'}),('unknown',{})]:
            try:
                execute(name,args)
            except ValueError as error:
                errors.append(str(error))
        try:
            strict_json('{"pages":3,"pages":30}')
        except ValueError as error:
            errors.append(str(error))
        return errors
    if code == '05-04':
        from .application_details import transport_comparison
        return transport_comparison()
    if code == '05-05':
        from .application_details import retrieval_evaluation, citation_examples
        return {'retrieval':retrieval_evaluation(),'citations':citation_examples()}
    if code in ['06-01','06-02','06-03','06-04','06-05']:
        from .runtime_details import exercise
        tmp=Path(__file__).resolve().parents[2]/'.tmp'
        tmp.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='solution-',dir=tmp) as work:
            return exercise(code,Path(work))
    raise ValueError('题号不在计算题索引中；第七篇参考推演见题解文件')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('code')
    args=parser.parse_args()
    print(json.dumps(solve(args.code),ensure_ascii=False,indent=2))
