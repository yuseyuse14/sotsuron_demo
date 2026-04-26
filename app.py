import streamlit as st
from pathlib import Path
import json
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import torch
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.express as px
import torch.nn.functional as F
import time


# ---ログ非表示用---
from transformers.utils import logging
logging.set_verbosity_error()


# ---定数---
MODEL_DIR = Path("./model")


# ---変数（保持）---
if "probs" not in st.session_state:
    st.session_state.probs = None


# ---関数---
@st.cache_resource
def load_data(dir):
    with open(dir / "word.json", "r", encoding="utf-8") as f:
        word_list = json.load(f)
    return word_list

@st.cache_resource
def load_model(dir):
    tokenizer = BertJapaneseTokenizer.from_pretrained(dir)
    model = BertForSequenceClassification.from_pretrained(dir)
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
        df = df.sort_values(by="確率", axis=1, ascending=False)
    elif sort == "確率が低い順に並び替え":
        df = df.sort_values(by="確率", axis=1, ascending=True)
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

def main():
    # データ読み込み
    words = load_data(dir=MODEL_DIR)
    tokenizer, model = load_model(dir=MODEL_DIR)

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
            else:
                st.session_state.probs = genre_predict(text_input, tokenizer=tokenizer, model=model)
        else:
            st.session_state.probs = genre_predict(text_option, tokenizer=tokenizer, model=model)

    if st.session_state.probs is not None:
        st.space()
        st.subheader("結果")
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