import gradio as gr

# ── 1.  Stubbed AI function (replace with your real model) ─────────────────
def generate_murals(img, prompt):
    # TODO: call your AI here using both img and prompt
    return [img, img, img, img]   # placeholder: four copies


# ── 2.  Build the Gradio UI ─────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Graffiti Mural Generator")

    # ── TOP ROW: upload on left, prompt+button on right ────────────────
    with gr.Row():
        with gr.Column(scale=1):
            upload_img = gr.Image(
                type="filepath",
                label="Upload Graffiti Wall"
            )
        with gr.Column(scale=1):
            theme   = gr.Textbox(
                placeholder="Enter mural theme (e.g. 'floral')",
                label="Mural Theme"
            )
            gen_btn = gr.Button(
                "Generate Murals",
                variant="primary"
            )

    # ── BOTTOM ROW: gallery of 4 generated murals side-by-side ───────────
    gallery = gr.Gallery(
        label="Your AI‐Generated Murals",
        columns=[4],         # four across
        show_label=False,
        height="auto"
    )

    # ── 3.  Wire it up ──────────────────────────────────────────────────────
    gen_btn.click(
        fn=generate_murals,
        inputs=[upload_img, theme],
        outputs=gallery
    )

# ── 4.  Launch ─────────────────────────────────────────────────────────────
demo.launch()



