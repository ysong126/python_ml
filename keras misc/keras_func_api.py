from keras.models import Model
from keras import layers
from keras import Input

import numpy as np

# multiple input model
def model1():
    text_vocabulary_size= 10000
    question_vocabulary_size=10000
    answer_vocabulary_size = 500

    text_input = Input(shape=(None,),dtype='int32',name='text')
    embedded_text = layers.Embedding(64,text_vocabulary_size)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)

    question_input = Input(shape=(None,),dtype='int32',name='question')
    embedded_question = layers.Embedding(32,question_vocabulary_size)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    concatenated = layers.concatenate([encoded_text,encoded_question],axis=-1)
    answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)

    model = Model([text_input,question_input],answer)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

    # training model
    num_samples= 1000
    max_length = 100
    text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
    question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
    answers = np.random.randint(0,1, size=(num_samples,answer_vocabulary_size))

    model.fit([text,question],answers,epochs=10,batch_size=128)
    model.fit({'text':text, 'question':question}, answers,epochs=10,batch_size=128)
    model.summary()

#multiple output model
def model2():
    vocabulary_size =50000
    num_income_groups = 10
    posts_input = Input(shape=(None,),dtype='int32',name='posts')
    embedded_posts = layers.Embedding(256,vocabulary_size)(post_input)
    x = layers.Conv1D(128,5,activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128,activation='relu')(x)
    age_prediction =layers.Dense(num_income_groups,activation='softmax',name='income')(x)
    income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
    gender_prediction = layers.Dense(1,activation='sigmoid',name='gender')(x)
    model = Model(posts_input, [age_prediction,income_prediction,gender_prediction])

    model.compile(optimizer='rmsprop',loss=['mse','categorical_crossentroy','binary_crossentropy'])
    model.compile(optimizer='rmsprop',loss={'age':'mse','income':'categorical_crossentropy','gender':'binary_crossentropy'})


    return