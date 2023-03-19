#Hackathon Online: NLU Intent Classification 
of Super AI Engineer Season 3

This is a competition of Super AI Engineer Season 3 on the topic of NLU Intent Classification. The following is a way to train a model and use it on the preliminary data provided by the competition on Github.

The techniques used for training the model include CountVectorizer, Tfidf, MLP, and VotingClassifier. 
The score used to evaluate the model is f1 and there are a total of 7 classes.

Install lib:

    !pip install onnx
    !pip install skl2onnx
    !pip install pythainlp
    !pip install onnxruntime

Loading the model:

    import onnxruntime as rt
    import numpy as np
    import pickle

    with open('/content/NLU_Intent_Classification.pkl', 'rb') as f:
        count_vect,dict_data = pickle.load(f)

    sess = rt.InferenceSession('/content/NLU_Intent_Classification.onnx')
	
Using the model:

    text = "show me the nearest movies at movie theatre for twenty one o clock"
    
    text_list = [text]
    input_data = count_vect.transform(text_list).toarray().astype('float32')
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_data})
    predicted_label = np.argmax(list(output[1][0].values()))
    print(dict_data[predicted_label])