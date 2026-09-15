# Handwritten Multi-Digit Text Recognition

This showcases how my program executes Each digit one by one by the trained model in "data" directory

<img width="492" height="300" alt="image" src="https://github.com/user-attachments/assets/01f4c5cd-d5cd-4242-bfae-f6cbab4c713c" />

<img width="441" height="206" alt="image" src="https://github.com/user-attachments/assets/fd4f912d-3837-40e5-8573-7819d8505530" />

# Usage
Write any number out in MS Paint or Freeform and take a screenshot of that
and it would give out a neat output just like the screenshot above


You can also use it in your project as the function returns the value read.

```
python3 predict.py <file-name>
```

# Train Your Own Model
In order to create your own model, I've shared a `mnist_classifier.py` file which trains model using pytorch

## Step 1:
Download any data set from the internet eg- Kaggle, and paste the raw files in a new folder of your working directory as `data\MNIST\raw`
## Step 2:
Set the `download` variable to False which means you have external dataset present
## Step 3: 
Run the code

and thats it!
