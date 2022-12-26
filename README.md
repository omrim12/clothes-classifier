# clothes-classifier
## Description
#### This application provides a CNN based AI model for identifying clothing items types.
#### This model is being pre-trained in runtime against the Tensorflow keras fashion MNIST dataset.

## Get started
#### Run the application in your local environment from git
1. Clone the repo:
```bash
git clone https://github.com/omrim12/clothes-classifier.git
```

2. Change into project directory:
```bash
cd clothes-classifier
```

3. Create a virtual environment:
```bash
python3 -m venv venv && source venv/bin/activate
```

*Note: this command requires an appropriate python version.*

4. Install application requirements:
```bash
pip3 install -r requirements.txt
```

*Note: this command requires an appropriate pip version.*

5. Run the application:
```bash
python3 src/clothes-classifier/clothes_driver.py
```

*Notes: for using `classify` tool, mount your cloth item image under the project dir*
#### Supported clothing types:
- T-shirt/top 
- Trouser
- Pullover 
- Dress 
- Coat 
- Sandal 
- Shirt 
- Sneaker 
- Bag 
- Ankle boot

### [Visit our GitHub page](https://github.com/omrim12/clothes-classifier)
### [Visit our PyPI page](https://test.pypi.org/project/clothes-classifier/1.0.2/)