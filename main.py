
import time
import numpy as np
import pandas as pd
import nltk
import string
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

import plotly.express as px

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

CLASS_NAMES = ["Fake", "Real"]
MAPPING_DICT = {
    "Fake":0,
    "Real":1
}

model_name = "BERTFakeNewsDetector"
MODEL_CALLBACKS = [ModelCheckpoint(model_name, save_best_only=True)]


fake_news_filepath = "/kaggle/input/fake-and-real-news-dataset/Fake.csv"
real_news_filepath = "/kaggle/input/fake-and-real-news-dataset/True.csv"

fake_df = pd.read_csv(fake_news_filepath)
real_df = pd.read_csv(real_news_filepath)

fake_df.head()


real_df.head()

real_df["Label"] = "Real"
fake_df["Label"] = "Fake"

df = pd.concat([fake_df, real_df])
df.reset_index()
df.head()

print(f"Dataset Size: {len(df)}")

data = df.sample(1000).drop(columns=["title", "subject", "date"])
data.Label = data.Label.map(MAPPING_DICT)
data.sample(10)

class_dis = px.histogram(
    data_frame = df,
    y = "Label",
    color = "Label",
    title = "Fake & Real Samples Distribution",
    text_auto=True
    )
class_dis.update_layout(showlegend=False)
class_dis.show()


subject_dis = px.histogram(
    data_frame = df,
    x = "subject",
    color = "subject",
    facet_col = "Label",
    title = "Fake & Real Subject Distribution",
    text_auto=True
    )
subject_dis.update_layout(showlegend=False)
subject_dis.show()
list(filter(lambda x: len(x)>20, df.date.unique()))

df = df[df.date.map(lambda x: len(x)) <= 20]
df.date = pd.to_datetime(df.date, format="mixed")
df.head()

label_date_hist = px.histogram(
    data_frame = df,
    x = 'date',
    color = "Label",
)
label_date_hist.show()

real_sub_hist = px.histogram(
    data_frame = df[df.Label == "Real"],
    x = 'date',
    color = "subject",
)
real_sub_hist.show()

subject_hist = px.histogram(
    data_frame = df,
    x = 'date',
    color = "subject",
)
subject_hist.show()

stop_words = set(stopwords.words('english'))
def text_processing(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    clean_text = ' '.join(filtered_words)
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation)).strip()
    return clean_text

X = data.text.apply(text_processing).to_numpy()
Y = data.Label.to_numpy().astype('float32').reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    train_size=0.9,
    test_size=0.1,
    stratify=Y,
    random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    train_size=0.9,
    test_size=0.1,
    stratify=y_train,
    random_state=42
)

bert_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(
    bert_name,
    padding = "max_length",
    do_lower_case = True,
    add_special_tokens = True,
)

X_train_encoded = tokenizer(
    X_train.tolist(),
    padding = True,
    truncation = True,
    return_tensors = "tf"
).input_ids

X_valid_encoded = tokenizer(
    X_valid.tolist(),
    padding = True,
    truncation = True,
    return_tensors = "tf"
).input_ids

X_test_encoded = tokenizer(
    X_test.tolist(),
    padding = True,
    truncation = True,
    return_tensors = "tf"
).input_ids

train_ds = tf.data.Dataset.from_tensor_slices((X_train_encoded, y_train)).shuffle(len(X_train)).batch(8).prefetch(tf.data.AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid_encoded, y_valid)).shuffle(len(X_valid)).batch(8).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test_encoded, y_test)).shuffle(len(X_test)).batch(8).prefetch(tf.data.AUTOTUNE)

bert_model = TFAutoModelForSequenceClassification.from_pretrained(bert_name, num_labels = 1)

bert_model.compile(
    optimizer = Adam(learning_rate = 1e-5),
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="Accuracy"),
        tf.keras.metrics.Precision(name="Precision"),
        tf.keras.metrics.Recall(name="Recall"),
    ]
)

model_history = bert_model.fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 5,
    batch_size = 16,
    callbacks = MODEL_CALLBACKS
)

model_history = pd.DataFrame(model_history.history)

bert_model.save(model_name)

import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss", "Accuracy", "Precision", "Recall"))

fig.add_trace(go.Scatter(y=model_history['loss'], mode='lines', name='Training Loss'), row=1, col=1)
fig.add_trace(go.Scatter(y=model_history['val_loss'], mode='lines', name='Validation Loss'), row=1, col=1)

fig.add_trace(go.Scatter(y=model_history['Accuracy'], mode='lines', name='Training Accuracy'), row=1, col=2)
fig.add_trace(go.Scatter(y=model_history['val_Accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=2)

fig.add_trace(go.Scatter(y=model_history['Precision'], mode='lines', name='Training Precision'), row=2, col=1)
fig.add_trace(go.Scatter(y=model_history['val_Precision'], mode='lines', name='Validation Precision'), row=2, col=1)

fig.add_trace(go.Scatter(y=model_history['Recall'], mode='lines', name='Training Recall'), row=2, col=2)
fig.add_trace(go.Scatter(y=model_history['val_Recall'], mode='lines', name='Validation Recall'), row=2, col=2)

fig.update_layout(
    title='Model Training History',
    xaxis_title='Epoch',
    yaxis_title='Metric Value',
    showlegend=False,
)

fig.update_xaxes(title_text='Epoch', row=1, col=1)
fig.update_xaxes(title_text='Epoch', row=1, col=2)
fig.update_xaxes(title_text='Epoch', row=2, col=1)
fig.update_xaxes(title_text='Epoch', row=2, col=2)

fig.update_yaxes(title_text='Loss', row=1, col=1)
fig.update_yaxes(title_text='Accuracy', row=1, col=2)
fig.update_yaxes(title_text='Precision', row=2, col=1)
fig.update_yaxes(title_text='Recall', row=2, col=2)

fig.show()


test_loss, test_acc, test_precision, test_recall = bert_model.evaluate(test_ds, verbose = 0)

print(f"Test Loss      : {test_loss}")
print(f"Test Accuracy  : {test_acc}")
print(f"Test Precision : {test_precision}")
print(f"Test Recall    : {test_recall}")

def predict_text(text, model):
    tokens = tokenizer(text, return_tensors = 'tf', padding="max_length", truncation=True).input_ids
    return np.abs(np.round(model.predict(tokens, verbose = 0).logits))

for _ in range(5):
    index = np.random.randint(len(X_test))
    
    text = X_test[index]
    true = y_test[index]
    model_pred = predict_text(text, model = bert_model)[0]
    
    print(f"ORGINAL TEXT:\n\n{text}\n\nTRUE: {CLASS_NAMES[int(true)]}\tPREDICTED: {CLASS_NAMES[int(model_pred)]}\n{'-'*100}\n")

