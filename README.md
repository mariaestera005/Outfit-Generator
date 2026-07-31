# Outfit-Generator-

👗***Outfit Generator*** is an original and intelligent outfit recommendation system based on image processing, linear algebra and Python that analyzes the dominant color of a clothing item using image processing and Singular Value Decomposition (SVD), then generates personalized outfit suggestions based on color harmony, style, season, and clothing attributes. To rank the recommendations, the system employs a mathematical model based on SVD and the Moore–Penrose pseudoinverse, allowing it to score and select the most suitable clothing combinations.

Each clothing item is stored in a JSON data file together with its key attributes, including:
  📍ID
  👕Category (Top, Bottom, Shoes, Accessories)
  🌤️ Season
  ✨ Style (Casual, Formal, Elegant, Sport)
  🎨 Dominant RGB color, added automatically by the algorithm

The recommendation engine then analyzes these features and generates complete outfit ideas by combining:

- Color compatibility
- Clothing category
- Seasonal suitability
- Style preferences
- User ratings and preferences.

## 🛠️ Technologies
- Python
- NumPy
- Matplotlib
- JSON
- Linear Algebra (SVD, Moore–Penrose Pseudoinverse)
- Image Processing
  
## 🚀 Project Goal

The objective of this project is to demonstrate how computer vision, linear algebra, and recommendation algorithms can be combined to build a smart fashion assistant capable of transforming a single clothing image into personalized outfit suggestions. Last but not least, this project provides an efficient solution for people who struggle to decide what to wear, helping them quickly choose well-matched outfits and look their best.

  
