# Text Summarizer
A transformer based Text Summarizer trained on about 36,000 Amazon Software Product Reviews

# Sample Conversions
```
Review: Product Key did not work!  Called Microsoft and they said the Key had already been activated.  Everything looks legit, nice sealed Microsoft envelope.  Win 10 DVD in another sleeve with Microsoft Licence Sticker.  I even had to peel of an official looking sticker that was covering the actual Product Key.
Summary: not worth the money
Review: I have used this product for many years on all my devices. It works well for me and is worry free. The price is not to bad also, and I think I'll be using it still in years to come
Summary: good product for the price
Input: Needed this to put on personal computer to do grad work. Using Google was not good enough.  LOVE the APA formatted paper made this purchase worth every penny!
Output: great for the price
Input: My program on CD would not install.  Intuit provides no support.  The site is set up to avoid questions.  There is no support on the web site for this kind of issue; and when I tried to speak to someone  I got an endless loop returning to the "options" list.  I will return the product and get something usable
Output: not compatible with anything
```

# How To Use
1. Install all the requirements by using the following code while in the current directory<br>
```pip install -r requirements.txt```<br>
2. Run ```python TextSummarizer.py```<br>
3. Done <br>

# Credits
1. Tensorflow's article on <a href="https://www.tensorflow.org/tutorials/text/transformer">transformers</a> <br>
2. Amazon's <a href="https://nijianmo.github.io/amazon/index.html">Product Reviews Dataset</a>