__all__ = ['is_cat', 'learn', 'classify_image', 'categories' , 'image', 'label', 'examples' , 'intf']

from fastai.vision.all import *
import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()

def is_cat(x): return x[0].isupper() 

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(shape=(192,192))
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
