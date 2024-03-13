mkdir glove
wget -P glove https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove/*zip -d glove
rm glove/*zip

unzip tweets.zip
rm tweets.zip