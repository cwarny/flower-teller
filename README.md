# Flower bot

## Downloading images

* `tail -n+2 flowers.csv | awk -F, '{print "http://www.image-net.org/download/synset?wnid="$1"&username=cwarny&accesskey=99f126958c2d6f2d295055abf3dc33f275951408&release=latest&src=stanford"}' | parallel wget -P ./n11669921`

## Creating directory structure for flowers

* `cd n11669921`
* `for f in *; do dir=$(echo $f | grep -Eo 'n[0-9]+'); mkdir $dir; tar -C $dir -xzf $f; rm $f; done`

## Training-validation split

* `python split.py`