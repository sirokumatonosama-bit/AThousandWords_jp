import gradio as gr
from typing import List

# ============================================================
# ここに本物のキャプション生成ロジックを後で入れる
# ============================================================

def generate_caption(image, model_name: str, prompt: str, negative_prompt: str) -> str:
    """
    本来はここで AThousandWords のモデルを呼び出す。
    今はダミーの英語キャプションを返す。
    """
    base = f"[Model: {model_name}] "
    if prompt:
        base += f"[Prompt: {prompt}] "
    if negative_prompt:
        base += f"[Negative: {negative_prompt}] "
    return base + "A sample caption describing the image."


# ============================================================
# 日本語翻訳（本番では DeepL / NLLB / M2M100 に差し替え）
# ============================================================

def translate_to_japanese(text: str) -> str:
    return "（日本語訳）画像を説明するサンプルキャプションです。"


# ============================================================
# 原文＋訳文の2段構成でキャプションを返す
# ============================================================

def caption_with_ja(
    image,
    model_name: str,
    prompt: str,
    negative_prompt: str,
    num_captions: int,
):
    if image is None:
        return ["画像が選択されていません。"]

    results = []
    for i in range(num_captions):
        eng = generate_caption(image, model_name, prompt, negative_prompt)
        jpn = translate_to_japanese(eng)

        combined = f"【原文】\n{eng}\n\n【訳文】\n{jpn}"
        results.append(combined)

    return results


# ============================================================
# Gradio GUI（日本語化＋原文/訳文2段表示）
# ============================================================

def build_interface():
    with gr.Blocks(title="A Thousand Words - 日本語版") as demo:
        gr.Markdown(
            """
# 🖼️ A Thousand Words（日本語版 GUI）

画像からキャプションを生成し、  
**上に英語（原文）、下に日本語訳（訳文）** を表示します。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="入力画像",
                    type="pil"
                )

                model_name = gr.Dropdown(
                    label="使用するモデル",
                    choices=[
                        "wd14",
                        "joycaption",
                        "florence2",
                        "qwen-vl",
                    ],
                    value="wd14",
                )

                prompt = gr.Textbox(
                    label="追加プロンプト（任意）",
                    placeholder="例：構図やスタイルの指定など",
                    lines=2
                )

                negative_prompt = gr.Textbox(
                    label="ネガティブプロンプト（任意）",
                    placeholder="例：除外したい要素など",
                    lines=2
                )

                num_captions = gr.Slider(
                    label="生成するキャプション数",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1
                )

                run_button = gr.Button("キャプション生成を開始")

            with gr.Column(scale=1):
                gr.Markdown("### 📝 生成結果（原文＋訳文）")
                captions_output = gr.JSON(
                    label="キャプション一覧"
                )

        run_button.click(
            fn=caption_with_ja,
            inputs=[image_input, model_name, prompt, negative_prompt, num_captions],
            outputs=[captions_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
