import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
import re
import numpy as np


# ---ログ非表示用---
from transformers.utils import logging
logging.set_verbosity_error()


# ---定数---
MODEL_NAME = "yuseyuse14/bert-japanese-genre-classifier"


# ---変数（保持）---
if "text" not in st.session_state:
    st.session_state.text = ""
if "probs" not in st.session_state:
    st.session_state.probs = None


# ---関数---
@st.cache_resource
def load_data(dir):
    # HuggingFaceから省略語データを取得
    file_path = hf_hub_download(
        repo_id=dir,
        filename="abbreviation.json",
        token=st.secrets["HF_TOKEN"]
    )
    abbr_df = pd.read_json(file_path)
    return abbr_df

@st.cache_resource
def load_model(dir):
    # HuggingFaceからモデルを取得
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

# 分野推定 -> 分野ごとの確率
def genre_predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.squeeze().numpy()
    return probs

def make_df(probs, sort, labels):
    soft_probs = temp_soft(probs, 1.5) * 100
    df = pd.DataFrame([soft_probs], index=["確率(%)"], columns=labels)
    df.index.name = "分野"
    df = df.round(2)

    if sort == "確率が高い順に並び替え":
        df = df.sort_values(by="確率(%)", axis=1, ascending=False)
    elif sort == "確率が低い順に並び替え":
        df = df.sort_values(by="確率(%)", axis=1, ascending=True)
    return df

def bar_graph(df):
    df_t = df.T
    fig = px.bar(
        df_t,
        x="確率(%)",
        y=df_t.index,
        orientation="h"
    )
    fig.update_layout(
        height=800,
        yaxis=dict(tickfont=dict(size=10))
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def temp_soft(x, temp=1.0):
    x = torch.tensor(x)
    return F.softmax(x / temp, dim=0).numpy()

def word_in_text(word, text):
    # 単語そのものが含まれるとき(PHPとかは反応しない、分かち書きは重い、、)
    pattern = rf"(?<![a-zA-Z]){re.escape(word)}(?![a-zA-Z])"
    return bool(re.search(pattern, text))

def js_divergence(P, Q, eps=1e-12):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    M = 0.5 * (P + Q)

    kl_pm = np.sum(P * np.log((P + eps) / (M + eps)))
    kl_qm = np.sum(Q * np.log((Q + eps) / (M + eps)))
    return 0.5 * (kl_pm + kl_qm)

def read_estimate(text, probs, abbr):
    filtered = abbr[abbr["word"].apply(lambda x: word_in_text(x, text))]
    if filtered.empty:
        return text
    else:
        scores = list()
        for _, row in filtered.iterrows():
            js = js_divergence(temp_soft(probs, 2.5), np.array(temp_soft(row["score"], 0.25)))
            scores.append(-js)
        filtered["js"] = scores
        max_row = filtered.loc[filtered["js"].idxmax()]
        target = max_row["word"]
        replacement = f" **{max_row['word']}({max_row['read']})** "
        return text.replace(target, replacement)

def main():
    # データ読み込み
    abbr = load_data(dir=MODEL_NAME)
    tokenizer, model = load_model(dir=MODEL_NAME)

    # UI
    st.title("B3 研究室見学")
    st.space()

    st.write("文章を入力すると、BERTを用いて分野と省略語の読みを推定します")
    st.write("推定できる省略語")
    st.write("・FF（ファイナルファンタジー/前輪駆動）")
    st.write("・HP（ホームページ/ヒットポイント）")

    text_option = st.selectbox(
        "例文",
        ["未選択",
        "FFシリーズの中だと、どの作品が一番好き？",
        "山道を走るなら、FFよりも4WDのほうが安心かもね",
        "公式HPにはそんな情報のってなかったな",
        "ボス戦でHPが1しか残っていなくて焦った"]
    )
    if text_option == "未選択":
        text_input = st.text_input("好きな文章を入力してください")

    if st.button("推定"):
        if text_option == "未選択":
            if text_input.strip() == "":
                st.warning("文章を入力してください")
                return
            else:
                st.session_state.text = text_input
        else:
            st.session_state.text = text_option
        st.session_state.probs = genre_predict(st.session_state.text, tokenizer=tokenizer, model=model)

    if st.session_state.probs is not None:
        st.space()
        st.subheader("結果")
        st.markdown(read_estimate(st.session_state.text, probs=st.session_state.probs, abbr=abbr))

        st.space()
        sort_option = st.selectbox(
            "並び替え",
            ["デフォルト",
            "確率が高い順に並び替え",
            "確率が低い順に並び替え"]
        )
        df = make_df(st.session_state.probs, sort=sort_option, labels=model.config.id2label.values())
        st.dataframe(df)
        fig = bar_graph(df)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()