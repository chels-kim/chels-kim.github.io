'''
Re-format all the residual code that remains from notebook-to-markdown conversion;
credits to Jake Tae (https://github.com/jaketae) and their post the blog
(https://jaketae.github.io/blog/jupyter-automation/)

Some portions were disabled because the tagger package couldn't be installed (apparently doesn't exist?).
Instead, I'm creating my own Front Matter yaml template.
'''

import re
import sys

from nbformat import NO_CONVERT, read

# from tagger.model import load_model


def rmd(nb):  # for R Markdown files
    with open(nb, "r") as file:
        filedata = file.read()
    filedata = re.sub('src="', 'src="/assets/img/', filedata)
    with open(nb, "w") as file:
        file.write(filedata)


def ipynb(nb):
    # title = nb.split(".")[0]
    # with open(f"{title}.ipynb") as f:
    #     notebook = read(f, NO_CONVERT)
    # text = get_text(notebook)
    # tags = predict_tags(text)
    yaml = build_yaml()
    with open(nb) as file:
        filedata = file.read()
    filedata = re.sub(r"!\[svg\]\(", '<img src="/assets/img/', filedata)
    filedata = re.sub(".svg\)", '.svg">', filedata)
    filedata = re.sub(r"!\[png\]\(", '<img src="/assets/img/', filedata)
    filedata = re.sub(".png\)", '.png">', filedata)
    filedata = yaml + filedata
    with open(nb, "w") as file:
        file.write(filedata)


# def get_text(notebook):
#     markdown_cells = [
#         cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
#     ]
#     text = " ".join(cell["source"] for cell in markdown_cells)
#     text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
#     text = re.sub(r"\$.*?\$", "math variable", text)
#     text = re.sub(r"\]\(.*?\)", r"]", text)
#     text = re.sub(r"(#)+ \w*", "", text)
#     text = text.replace("[", "").replace("]", "")
#     text = re.sub(r"`.*?`", "code variable", text)
#     text = text.replace("\n", " ")
#     text = " ".join(text.split())
#     return text


# def predict_tags(text):
#     model = load_model()
#     tags = model.predict(text)
#     return tags


def build_yaml():
    # tag_header = ""
    # for tag in tags:
    #     tag_header += f"  - {tag}\n"
    return (
        f"---\nlayout: post\ntitle: TITLE\nsubtitle: SUBTITLE\nmathjax: true\ntoc: true\n"
        f"author: Chelsea Kim\ntags: TAGS\n---\n\n"
    )


if __name__ == "__main__":
    opt = sys.argv[1]
    nb = sys.argv[2]
    if opt == "-r":
        rmd(nb)
    else:
        ipynb(nb)