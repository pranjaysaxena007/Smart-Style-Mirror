import streamlit as st
import tensorflow as tf
import numpy as np

# #Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("final_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


#SIDEBAR
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",['Home','About','Dress Recommendation'])

#Home Page
if(app_mode == 'Home'):
    st.header("SMART STYLE MIRROR")
    st.markdown("""<div style="text-align: justify";>

## AI-Powered Fashion Assistant for Men's Clothing

#### Overview

Smart Styling Mirror is an innovative system that combines artificial intelligence and deep learning to revolutionize personal styling. Designed for men, this intelligent mirror not only detects what you’re wearing but also provides smart suggestions on whether your outfit fits the occasion — be it a formal meeting, casual outing, festive celebration, or a trendy party.
    
### How It Works
1. **Upload Image:** Go to the **Dress Recommendation** page and upload your whole image with proper lighting condition.
2. **Analysis:** Our system will process the image using advanced algorithms to identify style tag.
3. **Results:** View the results and recommendations for further action.

#### Key Features

1. **Real-Time Outfit Detection**
Using a camera or uploaded image, the system identifies clothing items a person is wearing.

2. **Deep Learning-Based Clothing Analysis**
Our custom-trained CNN model classifies the clothing style into four main categories:

Casual

Formal

Funky

Traditional


3. **Smart Style Recommendation**
Based on the style tag and optional context (e.g., meeting, wedding), the system suggests whether the current outfit is suitable or not.

4. **User-Friendly Display**
The mirror displays the detected style and a recommendation message instantly — offering a seamless and stylish experience.

#### Try It Now

Step into the future of personal styling!
Let Smart Styling Mirror help you look perfect for every moment. cr

#### About Us
Learn more about the project, and our goals on the **About** page.
    </div>""", unsafe_allow_html=True
    )

#About Project
elif(app_mode == 'About'):
    st.header('About')
    st.markdown("""<div style="text-align: justify";>

#### About Dataset

To train and evaluate the deep learning model for the Smart Styling Mirror, a custom clothing dataset was created. The dataset includes a diverse set of male outfit images, carefully curated to represent four primary style categories:

Casual

Formal

Funky

Traditional


#### Data Collection Sources

**Online Resources**:
Images were collected from publicly available sources such as Google Images, fashion blogs, and style recommendation platforms.

**Self-Captured Photos**:
To increase authenticity and variability, several images were captured personally under different lighting conditions, poses, and backgrounds.

**Manual Tagging**:
Each image was manually labeled with one or more style tags based on clothing type, color, and context.


#### Dataset Highlights

**Total Images**:

Classes: 4 (Casual, Formal, Funky, Traditional)

Variations included:

Outfit combinations

Multiple backgrounds

Diverse lighting and angles
    </div>""", unsafe_allow_html=True
    )
#Prediction Page
elif(app_mode == 'Dress Recommendation'):
    st.header("Dress Recommendation")
    test_image = st.file_uploader('Choose an image:')
    #Show Image Button
    try:
        if(st.button('Show Image')):
            st.image(test_image, use_container_width=True)
    except:
        st.write("Please INSERT an Image for Viewing")    
    #Predict button
    if(st.button('Predict')):
        try:
            
            with st.spinner("Wait for it..."):
                st.markdown("""#### Our prediction""")
                result_index = model_prediction(test_image)
                # Define Class
                class_name = ['Casual', 'Formal', 'Funky', 'Traditional']
                st.success(f'''Model is Predicting it's a  ""{class_name[result_index]} outfit""''')
                st.snow()
        except:
            st.write("Please INSERT an Image for Prediction")
            
