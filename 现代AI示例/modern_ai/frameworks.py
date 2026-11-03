"""LangChain 组件与 LangGraph 检查点对照，业务约束仍复用原生实现。"""
import argparse
import json
import sys
from typing import TypedDict
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from .agent import execute, strict_json
from .retrieval import answer, embed, load_chunks, search
from .runtime import Store


class Minutes(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    pages: int = Field(ge=0, le=1000)
    questions: int = Field(ge=0, le=1000)


class CountEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # 框架接口只改变批量调用形式，坐标仍与原生计数完全相同。
        return [embed(text).tolist() for text in texts]

    def embed_query(self, text):
        return embed(text).tolist()


def component_mapping():
    chunks = load_chunks()
    documents = [Document(id=c['id'],page_content=c['text'],
                          metadata={k:c[k] for k in ['source','owner','revision','id']}) for c in chunks]
    index = InMemoryVectorStore.from_documents(documents, CountEmbeddings())
    pairs = index.similarity_search_with_score('切片共享',k=3)
    # 用同样证据形成实际提示对象，保留消息角色与正文。
    prompt = ChatPromptTemplate.from_messages([
        ('system','仅依据所附资料回答，保留来源。'),
        ('human','问题：{question}\n资料：{evidence}')])
    evidence = '\n'.join(f"[{d.metadata['id']}] {d.page_content}" for d,_ in pairs)
    messages = prompt.invoke({'question':'切片共享','evidence':evidence}).to_messages()
    parser = RunnableLambda(strict_json) | RunnableLambda(Minutes.model_validate)
    parsed = parser.invoke('{"pages":3,"questions":4}')
    # Store namespace 对应应用中的用户隔离；此内存 Store 不跨进程持久化。
    memory = InMemoryStore()
    namespace = ('users','alice')
    memory.put(namespace,'recovery',{'raw':'检查点保存状态。','summary':'保存状态'})
    return {'vector_results':[{'id':d.metadata['id'],'cosine':float(score)} for d,score in pairs],
            'messages':[{'role':m.type,'content':m.content} for m in messages],
            'parsed':parsed.model_dump(),'memory_value':memory.get(namespace,'recovery').value}


@tool(args_schema=Minutes)
def reading_minutes(pages: int, questions: int) -> dict:
    """按每页两分钟、每题一分钟估算阅读活动。"""
    # 框架负责解析分发，业务校验继续由同一个函数执行。
    return execute('reading_minutes', {'pages':pages,'questions':questions})


class State(TypedDict):
    task_id: str
    question: str
    result: dict
    approved: bool


class RetrievalState(TypedDict):
    question: str
    hits: list[dict]
    answer: str


def retrieval_graph():
    # 第五篇只用进程内状态，三个节点的返回值分别更新对应字段。
    def retrieve(state):
        return {'hits':search(state['question'])['selected']}

    def reply(state):
        return {'answer':'\n'.join(f"{c['text']} [{c['id']}]" for c in state['hits'])}

    def abstain(state):
        return {'answer':'资料中没有可引用的证据。'}

    builder = StateGraph(RetrievalState)
    builder.add_node('retrieve',retrieve)
    builder.add_node('reply',reply)
    builder.add_node('abstain',abstain)
    builder.add_edge(START,'retrieve')
    builder.add_conditional_edges('retrieve',lambda s:bool(s['hits']),
                                  {True:'reply',False:'abstain'})
    builder.add_edge('reply',END)
    builder.add_edge('abstain',END)
    return builder.compile()


def build_graph(checkpointer, store, fail_commit=None):
    def prepare(state):
        run = store.create(state['task_id'], state['question'])
        while run['status'] in ['created','prepared']:
            run = store.advance(state['task_id'])
        return {'result':run}

    def review(state):
        # 节点恢复时从头重入，interrupt 之前没有业务写入。
        approved = interrupt({'task_id':state['task_id'],
                              'proposal':state['result']['payload']['proposal']})
        if type(approved) is not bool:
            raise ValueError('确认值必须为布尔值')
        return {'approved':approved}

    def commit(state):
        # 该节点可能重入，Store 以 task_id 约束业务写入。
        return {'result':store.confirm(state['task_id'],fail=fail_commit,
                                       expected_version=state['result']['version'])}

    def cancel(state):
        return {'result':store.cancel(state['task_id'])}

    builder = StateGraph(State)
    builder.add_node('prepare', prepare)
    builder.add_node('review', review)
    builder.add_node('commit', commit)
    builder.add_node('cancel', cancel)
    builder.add_edge(START, 'prepare')
    builder.add_edge('prepare', 'review')
    builder.add_conditional_edges('review', lambda state:state['approved'],
                                  {True:'commit', False:'cancel'})
    builder.add_edge('commit', END)
    builder.add_edge('cancel', END)
    return builder.compile(checkpointer=checkpointer)


def graph_action(path, action, task_id='graph-17', thread_id=None, fail=None):
    # thread_id 选择图检查点；task_id 选择业务幂等键，两者显式分开。
    store = Store(path)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_graph(saver, store, fail_commit=fail)
        config = {'configurable':{'thread_id':thread_id or task_id}}
        snapshot=graph.get_state(config)
        if action == 'show':
            return {'next':list(snapshot.next),'values':snapshot.values,
                    'interrupts':[str(i) for task in snapshot.tasks for i in task.interrupts]}
        if action not in ['start','resume','cancel','continue']:
            raise ValueError('未知图操作')
        if action != 'start' and not snapshot.values:
            raise ValueError('没有可恢复的图线程')
        if action == 'start':
            value = {'task_id':task_id,'question':'检查点恢复'}
        elif action in ['resume','cancel']:
            value = Command(resume=action=='resume')
        else:
            value = None
        result = graph.invoke(value, config)
        # 中断对象转成可阅读的普通字典。
        return json.loads(json.dumps(result, ensure_ascii=False, default=str))


def demo():
    chain = RunnableLambda(answer)
    return {'chain':chain.invoke('切片共享'),
            'tool_schema':reading_minutes.args_schema.model_json_schema(),
            'tool_value':reading_minutes.invoke({'pages':3,'questions':4}),
            'component_mapping':component_mapping(),
            'graph_found':list(retrieval_graph().stream({'question':'切片共享'},stream_mode='updates')),
            'graph_missing':list(retrieval_graph().stream({'question':'火星天气'},stream_mode='updates'))}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['demo','start','resume','cancel','continue','show'])
    parser.add_argument('--db')
    parser.add_argument('--id',default='graph-17')
    parser.add_argument('--thread-id')
    parser.add_argument('--fail',choices=['before_commit','after_commit'])
    args = parser.parse_args()
    if args.action != 'demo' and not args.db:
        parser.error('图操作需要 --db')
    print(json.dumps(demo() if args.action == 'demo' else graph_action(
                         args.db,args.action,args.id,args.thread_id,args.fail),
                     ensure_ascii=False, indent=2))
