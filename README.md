#Hackathon Online: NLU Intent Classification 
of Super AI Engineer Season 3

This is a competition of Super AI Engineer Season 3 on the topic of NLU Intent Classification. The following is a way to train a model and use it on the preliminary data provided by the competition on Github.

The techniques used for training the model include CountVectorizer, Tfidf, MLP, and VotingClassifier. 
The score used to evaluate the model is f1 and there are a total of 7 classes.

Loading the model:

    import pickle
    with open('0.96NLU_Intent_Classification_Super_AI_EngineerSS3.pkl', 'rb') as f:
        count_vect,tf_transformer,dict_data,model = pickle.load(f)
    
	
Using the model:

    data_precidt = [text_list_input]
    CtV = count_vect.transform(data_precidt)
    tfidf = tf_transformer.transform(CtV)
    list_data_predict = ANN.predict(tfidf)
    list_data_predict