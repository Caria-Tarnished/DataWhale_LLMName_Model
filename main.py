import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random
import re

# 加载向量嵌入模型和Yuan2大模型
@st.cache_resource
def get_models():
    # 加载Yuan2模型
    tokenizer = AutoTokenizer.from_pretrained('IEITYuan/Yuan2-2B-Mars-hf/', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-2B-Mars-hf/', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

    # 加载嵌入模型
    embedding_model = SentenceTransformer('AI-ModelScope/bge-small-zh-v1.5')
    
    # 加载文档嵌入和原始文档
    document_embeddings = np.load('poetry_embeddings.npy')
    with open('poetry_corpus.txt', 'r', encoding='utf-8') as f:
        documents = f.read().splitlines()
    
    return tokenizer, model, embedding_model, document_embeddings, documents

# 定义检索函数
def retrieve_relevant_documents(query, embedding_model, document_embeddings, documents, top_k=3):
    query_embedding = embedding_model.encode([query])
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# 主应用逻辑
def main():
    st.set_page_config(page_title="文墨启名", page_icon="📜", layout="wide")
    
    def reset_session_state():
        for key in st.session_state.keys():
            del st.session_state[key]

    # 回调函数，用于在输入框变化时更新 session state
    def update_user_input():
        st.session_state.user_input = st.session_state.input_temp
        st.session_state.step = 1

    def update_name_expectation():
        st.session_state.name_expectation = st.session_state.name_expectation_temp
        st.session_state.step = 2

    # 侧边栏
    st.sidebar.title("操作")
    if st.sidebar.button("开启新对话"):
        reset_session_state()
        st.experimental_rerun()

    # 示例用例展示
    st.sidebar.title("示范用例")
    st.sidebar.write("点击以下示例快速生成名字：")
    examples = ["李", "王", "张", "赵", "刘"]
    selected_example = st.sidebar.selectbox("选择一个示例姓氏", examples)

    st.title("📜 文墨启名")
    st.write("通过《诗经》、《楚辞》、《唐诗三百首》、《宋词》等古典诗词，为宝宝取一个有文化底蕴的名字。")

    st.markdown("""
        <style>
        body {
            background-image: url('https://pic.chinesefontdesign.com/uploads/2017/03/chinesefontdesign.com_2017-03-20_17-52-49.jpg');
            background-size: cover;
        }
        .stApp {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # 初始化对话状态
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.user_input = selected_example
        st.session_state.name_expectation = ""

    # 用户输入姓氏
    st.text_input("请输入宝宝的姓氏", key="input_temp", value=st.session_state.user_input, on_change=update_user_input)

    if st.session_state.step == 1:
        st.write("您希望名字有什么特别的含义或风格？（例如：优雅、富有诗意等）")
        st.text_input("名字期望", key="name_expectation_temp", on_change=update_name_expectation)

    # 添加一个按钮来触发名字生成逻辑
    if st.session_state.step == 2:
        generate_name_button = st.button("生成名字", key="generate_name_button")
        if generate_name_button:
            st.write("正在生成名字和解释...")
            with st.spinner('请稍候...'):
                # 加载模型和数据
                tokenizer, model, embedding_model, document_embeddings, documents = get_models()
                
                # 结合用户输入和期望
                full_input = f"我孩子的姓是{st.session_state.user_input}。用户期望：{st.session_state.name_expectation}"

                # 检索相关文档
                retrieved_docs = retrieve_relevant_documents(full_input, embedding_model, document_embeddings, documents)

                # 随机选取部分文档来增加多样性
                selected_docs = random.sample(retrieved_docs, min(3, len(retrieved_docs)))

                # 构建生成提示词
                prompt = f"姓氏：{st.session_state.user_input}\n期望：{st.session_state.name_expectation}\n相关古典诗词内容：\n"
                for doc in selected_docs:
                    prompt += f"- {doc}\n"
                prompt += "生成一个适合的名字，并解释其含义："

                # 调用生成模型
                inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7, top_p=0.9)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # 处理生成的输出
                final_name = re.sub(r"(姓氏：|期望：|相关古典诗词内容：|生成一个适合的名字，并解释其含义：)", "", response).strip()

                if final_name:
                    st.success("生成完毕！")
                    st.write("### 为您生成的名字和解释如下：")
                    st.write(final_name)
                else:
                    st.error("未能生成合适的名字和解释，请稍后重试。")

        # 重置对话状态
        if st.sidebar.button("重新开始"):
            reset_session_state()
            st.experimental_rerun()

if __name__ == '__main__':
    main()
