# dc_classifier
ACCURACY: 82% (improving)<br>
<p style="color:red">A simple dogs and cats classifier made with Keras.</p>

<h1>DEPENDENCIES:</h1>

pip install numpy<br>
pip install keras<br>
<br>
If you want train or optimize the model uncomment create_model command in prediction.py (and comment the load_model command) and download dogs and cats dataset.


<h1>TUTORIAL:</h1>

1) Download the project and extract it.<br>
2) Edit predictions.py adding in path variable the image path (containing dog(s) or cat(s))<br>
3) Run the program and get in result[0][0] the predicted value: 0 - Cat, 1 - Dog.
