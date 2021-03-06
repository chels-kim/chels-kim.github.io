#!/bin/sh
# Convert notebook to markdown, move file to _posts folder and related image files to the assets/images folder, if any.
# Credits to Jake Tae (https://github.com/jaketae) and their post the blog
# (https://jaketae.github.io/blog/jupyter-automation/)


nb=$1


function ipynb(){
	py format.py "-p" $nb

	mv ${nb%.ipynb}.md ../_posts/
    if compgen -G "*.png" > /dev/null
    then
        echo "==========Moving image files=========="
        mkdir ../assets/img/${nb%.ipynb}_files
        mv *.png ../assets/img/${nb%.ipynb}_files
    fi
}


function rmd(){
    eval "$(conda shell.bash hook)"
    conda activate R
    R -e "rmarkdown::render('$nb')"
    py format.py "-r" ${nb%.Rmd}.md
    mv ${nb%.Rmd}.md ../_posts/
    if [[ -d ${nb%.Rmd}_files ]]
    then
        echo "==========Moving image files=========="
        mv ${nb%.Rmd}_files ../assets/img/
    fi
}


function format(){
    if [[ $nb == *.ipynb ]]
    then
        echo "==========Starting .ipynb conversion=========="
        ipynb
    else
        echo "==========Starting .Rmd formatting=========="
        rmd
    fi
    echo "==========Formatting complete!=========="
    read -p "Press enter to continue"
    cd ../_posts
    open ${nb%.*}.md
}


format