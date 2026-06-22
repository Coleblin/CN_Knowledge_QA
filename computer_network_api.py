import os
import re
import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn

# 可选：使用OpenAI API提升回答质量
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI库未安装，将使用智能匹配模式回答")

# ==================== 数据模型定义（仅需要question字段，完全匹配Java端调用） ====================
class QueryRequest(BaseModel):
    """查询请求模型 - 仅接收question字段"""
    question: str = Field(..., description="用户的问题")

class QueryResponse(BaseModel):
    """查询响应模型 - 完全适配Java端期望格式"""
    code: int = 200
    msg: str = "success"
    data: Dict[str, str] = Field(default_factory=lambda: {"answer": ""})

# ==================== RAG引擎实现 ====================
@dataclass
class DocumentChunk:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict

class ComputerNetworkRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"正在加载嵌入模型: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix: np.ndarray = None
        self.knowledge_base_path = "computer_network_kb.txt"

    def smart_text_splitter(self, text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[str]:
        text = text.replace('\r\n', '\n').strip()
        sections = re.split(r'(?=## [一二三四五六七八九十]+、|### \d+\.\d+)', text)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        current_chunk = ""

        for section in sections:
            if '|' in section and '\n|' in section:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(section.strip())
                continue

            paragraphs = re.split(r'\n\s*\n', section)
            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def build_knowledge_base(self, knowledge_text: str = None) -> None:
        if knowledge_text is None:
            if not os.path.exists(self.knowledge_base_path):
                self._create_computer_network_kb()
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_text = f.read()

        print("正在进行智能文本分块...")
        chunk_texts = self.smart_text_splitter(knowledge_text)

        print("正在生成向量嵌入...")
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        self.chunks = []
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            self.chunks.append(DocumentChunk(
                id=f"chunk_{i}",
                content=text,
                embedding=embedding,
                metadata={"source": "computer_network_kb", "chunk_id": i}
            ))

        self.embeddings_matrix = np.array([chunk.embedding for chunk in self.chunks])
        print(f"知识库构建完成！共 {len(self.chunks)} 个文档块")

    def _create_computer_network_kb(self) -> None:
        print("创建计算机网络知识库文件...")
        kb_content = """
计算机网络知识点

## 一、计算机网络基础概述

### 1.1 计算机网络定义

将地理位置分散、具有独立功能的多台计算机，通过通信设备和传输介质连接，由网络软件实现资源共享和数据通信的计算机系统。

### 1.2 网络两大核心功能

• 数据通信：实现主机之间的数据传输（微信聊天、网页访问）

• 资源共享：硬件共享（打印机）、软件共享、数据共享（云盘）

### 1.3 网络分类（常考）

1. 按覆盖范围划分

￮ 局域网LAN：覆盖范围小（一栋楼/校园），速率高，延迟低

￮ 城域网MAN：覆盖一座城市

￮ 广域网WAN：跨城市/国家，互联网属于最大的广域网，速率低、延迟高

2. 按传输方式划分：点对点网络、广播式网络

3. 按拓扑结构划分：星型（主流）、总线型、环型、网状型

### 1.4 网络性能指标（必考）

• 带宽：网络最高数据传输速率，单位bit/s（注意区分字节B，1B=8bit）

• 时延：数据从发送到接收总耗时，分为4类
       

a. 发送时延：数据推送到信道的时间

b. 传播时延：信号在信道介质中传输的时间（只和距离、介质有关）

c. 处理时延：路由器交换机解析数据包耗时

d. 排队时延：数据包在路由器缓存排队等待转发的时间（网络拥堵主要原因）

• 丢包率：网络拥堵时数据包丢失的概率

• 吞吐量：单位时间内实际通过网络的数据量

## 二、网络分层体系（重中之重：OSI七层 & TCP/IP五层模型）

### 2.1 分层核心思想

下层为上层提供服务，上层调用下层服务；层与层之间不能跨层通信；对等层之间通过协议通信。

### 2.2 七层OSI模型（理论标准）VS 五层TCP/IP模型（工业实际使用）

|层级|OSI七层|TCP/IP五层（考试通用）|核心作用|典型协议/设备|
|---|---|---|---|---|
|7|应用层|应用层|面向用户，提供网络服务|HTTP、HTTPS、FTP、DNS、SMTP|
|6|表示层|<br/>|数据加密、解密、格式转换、编码转换|SSL/TLS|
|5|会话层|<br/>|建立、管理、终止会话连接|无常用协议|
|4|传输层|传输层|端口寻址，端到端通信，流量控制|TCP、UDP；网关|
|3|网络层|网络层|IP寻址，路由选择，分组转发|IP、ICMP、ARP；路由器|
|2|数据链路层|数据链路层|MAC寻址，封装成帧，差错检测|以太网协议；交换机、网桥|
|1|物理层|物理层|比特流传输，电气信号转换|网线、光纤、集线器|

### 2.3 数据封装与解封装（必考流程）

1. 发送方：应用层数据 → 传输层加TCP/UDP首部 → 网络层加IP首部 → 链路层加MAC首部尾部 → 物理层转为比特流发送

2. 接收方：物理层接收比特流 → 逐层剥去首部 → 最终还原原始应用数据

|核心口诀：上到下逐层加头，下到上逐层去头|
|---|

## 三、五层模型各层核心知识点（逐层详解）

### 3.1 物理层

• 传输单位：比特（bit）

• 核心功能：透明传输比特流，不做差错校验

• 常见传输介质：双绞线（100m限制）、光纤（远距离、抗干扰）、无线电磁波

• 设备：集线器（一层设备，无脑广播，无寻址能力）

### 3.2 数据链路层

• 传输单位：帧（Frame）

• 核心功能：成帧、MAC地址寻址、CRC差错检测、透明传输

• MAC地址：物理地址，48位，全球唯一，固化在网卡，二层寻址

• 设备：交换机（二层设备，基于MAC转发，隔离冲突域）

• 两大子层：LLC逻辑链路控制子层、MAC介质访问控制子层

### 3.3 网络层（IP核心层）

• 传输单位：分组/数据包

• 核心功能：IP逻辑寻址、路由选择、异构网络互联

• IP地址：32位IPv4，逻辑地址，可修改；分为网络号+主机号

• IP地址分类：A/B/C/D/E类，日常使用A、B、C三类
        

￮ A类：1-126，大型网络，默认子网掩码255.0.0.0

￮ B类：128-191，中型网络，默认子网掩码255.255.0.0

￮ C类：192-223，小型局域网，默认子网掩码255.255.255.0

• 核心协议：
        

a. ARP：地址解析协议，IP地址 → MAC地址

b. RARP：反向地址解析，MAC → IP

c. ICMP：网际控制报文协议，用于网络探测（ping命令底层协议）

• 设备：路由器（三层设备，基于IP转发，隔离广播域+冲突域）

### 3.4 传输层（TCP&UDP核心考点）

• 传输单位：报文段

• 核心功能：端口寻址，进程到进程通信，弥补网络层不可靠传输

• 端口号：0-65535，熟知端口0-1023
        

￮ HTTP：80  HTTPS：443  FTP：21  DNS：53

#### TCP与UDP对比（高频面试/考题）

|对比维度|TCP（传输控制协议）|UDP（用户数据报协议）|
|---|---|---|
|连接特性|面向连接，传输前建立连接|无连接，直接发送数据|
|可靠性|可靠传输，无丢失、无重复、有序|不可靠传输，可能丢包、乱序|
|拥塞/流量控制|具备流量控制、拥塞控制|无任何控制机制|
|首部开销|最小20字节，开销大|固定8字节，开销极小|
|适用场景|文件传输、网页、邮件（要求可靠）|直播、语音通话、游戏（要求实时）|

#### TCP三大核心机制（面试必问）

1. 三次握手（建立连接）为什么需要三次握手，不能两次？：防止失效的连接请求报文段占用服务端资源

￮ 第一次握手：客户端发SYN报文，客户端进入同步已发送状态

￮ 第二次握手：服务端收到SYN，返回SYN+ACK报文，服务端进入同步收到状态

￮ 第三次握手：客户端收到确认，发送ACK报文，双方连接建立，进入已连接状态

2. 四次挥手（断开连接）：TCP是全双工通信，双方都需要单独关闭发送通道，因此需要四次报文交互

3. 流量控制&拥塞控制

￮ 流量控制：控制发送方速率，避免接收方缓冲区溢出（点对点）

￮ 拥塞控制：控制全局网络发送速率，避免整个网络拥堵（全局网络）

### 3.5 应用层

• 传输单位：报文

• 核心功能：直接为用户应用进程提供网络服务

• 常用应用层协议详解：
        

a. HTTP：超文本传输协议，明文传输，无状态，端口80

b. HTTPS：HTTP+SSL/TLS加密，密文传输，安全，端口443

c. DNS：域名解析协议，域名解析为IP，端口53，基于UDP

d. FTP：文件传输协议，端口21控制端口，20数据端口

e. DHCP：动态主机配置协议，自动分配IP地址

## 四、常见网络设备对比（考试易混淆）

|设备|工作层级|寻址依据|冲突域|广播域|核心特点|
|---|---|---|---|---|---|
|集线器|物理层|无寻址|不隔离|不隔离|所有端口广播，性能最差|
|交换机|数据链路层|MAC地址|隔离|不隔离|单端口单独冲突域，全域广播|
|路由器|网络层|IP地址|隔离|隔离|既隔离冲突域也隔离广播域，连接不同网段|

## 五、HTTP&HTTPS 高频考点

### 5.1 HTTP协议特点

• 无状态：服务器不保存客户端历史访问信息（使用Cookie/Session解决无状态问题）

• 明文传输，数据不安全

• 基于TCP协议

### 5.2 HTTP请求报文结构

请求行（方法+URL+版本）→ 请求头 → 空行 → 请求体

### 5.3 常见HTTP请求方法

• GET：查询数据，参数拼接在URL，不安全，有长度限制

• POST：提交数据，参数在请求体，安全，无长度限制

• PUT：修改资源；DELETE：删除资源；HEAD：只获取响应头

### 5.4 HTTPS加密流程

混合加密：非对称密钥协商 + 对称密钥传输数据；同时加入数字证书防止中间人攻击

## 六、网络安全基础知识点

• 中间人攻击：拦截客户端与服务端通信数据，HTTPS证书可防御

• DoS/DDoS攻击：拒绝服务攻击，海量请求打垮服务器，导致无法正常响应

• 防火墙：部署在网络边界，基于IP/端口过滤非法报文

• VPN：虚拟专用网络，在公网搭建加密私有通道

## 七、期末必考简答题汇总（直接背诵）

1. 简述TCP三次握手过程及目的

2. TCP和UDP的区别以及各自适用场景

3. 集线器、交换机、路由器的区别

4. HTTP和HTTPS的区别

5. ARP协议的作用是什么

6. 网络时延分为哪四类，分别说明含义

## 八、易混淆知识点避坑总结

• 字节和比特：1字节(Byte)=8比特(bit)，带宽单位是bit/s，文件大小单位是Byte

• MAC地址物理不变，IP地址逻辑可变

• 交换机不能隔离广播风暴，路由器可以隔离广播风暴

• 三次握手建立连接，四次挥手断开连接，挥手次数多是因为TCP全双工
"""
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            f.write(kb_content)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.25) -> List[DocumentChunk]:
        if not self.chunks:
            raise ValueError("知识库尚未构建")

        query_embedding = self.embedding_model.encode(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((self.chunks[idx], similarities[idx]))

        results.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in results]

    def generate_answer(self, query: str, context: str, use_llm: bool = True) -> str:
        if not context:
            return "抱歉，我在计算机网络知识库中没有找到与您问题相关的信息。"

        if not OPENAI_AVAILABLE or not use_llm:
            return f"根据计算机网络知识库内容，为您找到以下相关信息：\n\n{context}"

        try:
            client = OpenAI()
            system_prompt = """你是一个专业的计算机网络课程助教。请严格基于以下提供的上下文回答用户的问题。
如果上下文中没有相关信息，请明确说明"抱歉，我在计算机网络知识库中没有找到与您问题相关的信息。"
绝对不要编造任何不在上下文中的内容。回答要简洁、准确、有条理，适合考试复习使用。
如果用户的问题与计算机网络无关，请回答"抱歉，我只能回答与计算机网络课程相关的问题。"
对于对比类问题，请尽量使用表格形式呈现。对于流程类问题，请分步骤说明。
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"上下文：\n{context}\n\n问题：{query}"}
                ],
                temperature=0.05,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用失败，使用智能匹配模式：{e}")
            return f"根据计算机网络知识库内容，为您找到以下相关信息：\n\n{context}"

    def process_query(self, request: QueryRequest) -> QueryResponse:
        try:
            relevant_chunks = self.retrieve(request.question)
            context = "\n\n---\n\n".join([chunk.content for chunk in relevant_chunks])
            answer = self.generate_answer(request.question, context)
            return QueryResponse(data={"answer": answer})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理查询失败: {str(e)}")

# ==================== FastAPI应用 ====================
app = FastAPI(title="计算机网络知识库问答API", version="1.0.0")
rag_engine: Optional[ComputerNetworkRAG] = None

@app.on_event("startup")
async def startup_event():
    global rag_engine
    rag_engine = ComputerNetworkRAG()
    rag_engine.build_knowledge_base()
    print("="*60)
    print("✅ API服务启动成功！")
    print("📍 接口地址：POST http://127.0.0.1:8000/api/qa")
    print("📚 接口文档：http://127.0.0.1:8000/docs")
    print("🏥 健康检查：http://127.0.0.1:8000/health")
    print("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    global rag_engine
    rag_engine = None
    print("API服务已关闭")

def get_rag_engine() -> ComputerNetworkRAG:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG引擎未就绪")
    return rag_engine

# 最终修正：接口路径 /api/qa，仅接收question字段
@app.post("/api/qa", response_model=QueryResponse, summary="查询计算机网络知识库")
async def query_knowledge_base(
        request: QueryRequest,
        rag: ComputerNetworkRAG = Depends(get_rag_engine)
):
    return rag.process_query(request)

@app.get("/health", summary="健康检查接口")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time() * 1000)}

if __name__ == "__main__":
    uvicorn.run(
        "computer_network_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )