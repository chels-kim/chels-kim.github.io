'''
Re-format all the residual code that remains from notebook-to-markdown conversion;
credits to Jake Tae (https://github.com/jaketae) and their post the blog
(https://jaketae.github.io/blog/jupyter-automation/)

Some portions were disabled because the tagger package couldn't be installed (apparently doesn't exist?).
Instead, I'm creating my own Front Matter yaml template.
'''

import re
import sys
import os
from nbconvert.preprocessors import TagRemovePreprocessor
from nbconvert.writers import FilesWriter
from nbconvert.exporters import MarkdownExporter
from traitlets.config import Config

from nbformat import NO_CONVERT, read

# from tagger.model import load_model

# current working directory
cwd = os.getcwd()

def rmd(nb):  # for R Markdown files
    with open(nb, "r") as file:
        filedata = file.read()
    filedata = re.sub('src="', 'src="/assets/img/', filedata)
    with open(nb, "w") as file:
        file.write(filedata)


def ipynb(nb):
    # Remove parts of cell:

    c = Config()

    # Configure our tag removal
    c.TagRemovePreprocessor.remove_cell_tags = ('hide-cell',)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('hide-output',)
    c.TagRemovePreprocessor.remove_input_tags = ('hide-input',)
    c.TagRemovePreprocessor.enabled = True

    # Configure and run our exporter
    c.MarkdownExporter.preprocessors = ['nbconvert.preprocessors.TagRemovePreprocessor']

    exporter = MarkdownExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

    body, resources = exporter.from_filename(nb)

    # Add tags to markdown
    # title = nb.split(".")[0]
    # with open(f"{title}.ipynb") as f:
    #     notebook = read(f, NO_CONVERT)
    # text = get_text(notebook)
    # tags = predict_tags(text)
    yaml = build_yaml()

    # Get filenmae
    filename = os.path.splitext(nb)[0]

    # Clean up the Markdown
    body = re.sub(r"!\[svg\]\(", '<img src="/assets/img/'+filename+'_files/', body)
    body = re.sub(".svg\)", '.svg">', body)
    body = re.sub(r"!\[png\]\(", '<img src="/assets/img/'+filename+'_files/', body)
    body = re.sub(".png\)", '.png">', body)
    body = yaml + body

    # Save files
    writer = FilesWriter()
    writer.write(body, resources, filename)


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
        f"---\nlayout: posts\ntitle: TITLE\nsubtitle: SUBTITLE\nmathjax: true\ntoc: true\n"
        f"author: Chelsea Kim\ntags: TAGS\n---\n\n"
    )


if __name__ == "__main__":
    opt = sys.argv[1]
    nb = sys.argv[2]
    # opt = '-p'
    # nb = '2021-03-22-covid-clinical-trials.ipynb'
    if opt == "-r":
        rmd(nb)
    else:
        ipynb(nb)
