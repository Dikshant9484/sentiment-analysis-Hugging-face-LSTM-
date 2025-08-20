import gradio as gr
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras.preprocessing.sequence import pad_sequences


max_features = 15000
maxlen = 200
embedding_dim = 32

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

def text_to_sequence(text):
    tokens = text.lower().split()
    sequence = [1]  
    for word in tokens:
        idx = word_index.get(word, 2)  
        sequence.append(idx)
    return pad_sequences([sequence], maxlen=maxlen, padding='post')

def predict_sentiment(text):
    seq = text_to_sequence(text)
    pred = model.predict(seq)[0][0]
    sentiment = "Positive 😊" if pred >= 0.5 else "Negative 😢"
    return sentiment

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review here..."),
    outputs=gr.Label(label="Sentiment"),
    title="IMDB Movie Review Sentiment",
    description="Enter a movie review and the model will predict whether it is positive or negative."
)

if __name__ == "__main__":
    demo.launch()
