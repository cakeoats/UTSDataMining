import streamlit as st
import cv2
import numpy as np
import requests
import matplotlib
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Set Matplotlib to use a non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function to load and preprocess the image from a URL
@st.cache_data
def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from URL.")
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image_file):
    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        if img is None:
            raise ValueError("Failed to decode uploaded image.")
        img_resized = cv2.resize(img, (256, 256))  # Resize for efficiency
        img_normalized = img_resized / 255.0  # Normalize
        return img_resized, img_normalized
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        return None, None

# Function to extract features (color and texture)
def extract_features(image, use_lbp=False):
    # Ensure the image is in the correct depth format (uint8)
    image_uint8 = (image * 255).astype(np.uint8)
    
    features = image_uint8.reshape(-1, 3)
    if use_lbp:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_features = lbp.reshape(-1, 1)
        features = np.hstack((features, lbp_features))
    return StandardScaler().fit_transform(features)

# Function to perform clustering using K-Means
def perform_clustering(features, n_clusters, max_iter, init_method):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init_method)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

# Function to calculate silhouette score
def calculate_silhouette_score(features, labels):
    return silhouette_score(features, labels)

# Function to visualize clustering results
def visualize_clustering(image, labels, n_clusters):
    clustered_image = labels.reshape(image.shape[:2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(clustered_image, cmap='viridis')
    ax2.set_title(f"Clustered Image (K={n_clusters})")
    ax2.axis('off')
    return fig

# Set up the Streamlit app
st.title("Aplikasi Clusterisasi Gambar Citra Udara Menggunakan Metode K-Means")


# Menggunakan fungsi HTML khusus (untuk HTML lebih kompleks)
html_code = """
<div style="background-color:transparant ;color:white">
    <p>140810210039_Ibrahim Dafi Iskandar.</p>
    <p>140810210048_Akmal Azzary Megaputra.</p>
    <p>140810210049_Lazia Firli Adlisnandar.</p>
</div>
"""
st.components.v1.html(html_code, height=100)

# Provide options for default images and uploaded image
image_source = st.radio("Select Image Source", ["Default images", "Upload your own image"])

# Dictionary of default images with display name as key and URL as value
default_images_urls = {
    'Dubai': 'https://c.pxhere.com/images/67/55/4d727e8eb6438a59a0b695489162-1425415.jpg!d',
    'Luxury Pool': 'https://www.shutterstock.com/image-photo/capture-elegance-luxury-pool-backyard-600nw-2507869465.jpg',
    'Forest View': 'https://asset-a.grid.id//crop/0x0:0x0/700x465/photo/2022/09/17/klasifikasi-macam-foto-udarajpg-20220917082702.jpg',
    'Bandara Jakarta': 'https://asset-2.tstatic.net/jateng/foto/bank/images/ara-mengunduh-gambar-dari-google-earth-view-langkah-kedua.jpg',
    'Pemukiman dan kota': 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgliSY1lUK8-QtYDesALTM_2PSrc56QnUQePfKqlP1D2PKmCWjREBklHc4J3OdOpYWffhdRSOiDNHP2FO2Sg_rgzyYvdwyXVe01is2Qk8j6Q_P97wmNLrR7uVRvQif3ISuKg39jFLKUgMw/s1600/Citra+Foto.jpg'
}

image = None  # Initialize image variable
normalized_image = None

# If user selects default images, show the selection option
if image_source == "Default images":
    selected_image_name = st.selectbox("Select a default image", list(default_images_urls.keys()))
    selected_image_url = default_images_urls[selected_image_name]
    image = load_image_from_url(selected_image_url)
    if image is not None:
        st.image(image, channels="BGR", caption=f"Selected Image: {selected_image_name}", use_column_width=True)
        normalized_image = image / 255.0  # Normalize default image if selected

# If user chooses to upload their own image
elif image_source == "Upload your own image":
    uploaded_image = st.file_uploader("Upload aerial image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image, normalized_image = load_and_preprocess_image(uploaded_image)
        if image is not None:
            st.image(image, channels="BGR", caption="Uploaded Aerial Image", use_column_width=True)

# Clustering parameters
n_clusters = st.slider("Jumlah Klustersiasi yang di inginkan", 2, 10, 3)
max_iter = st.slider("Maximal Iterasi Untuk K-Means", 100, 1000, 300)
init_method = st.selectbox("K-Means initialization method", ['k-means++'])
use_lbp = st.checkbox("Gunakan fitur textur")

# Perform clustering if an image is selected or uploaded
if normalized_image is not None:
    # Extract features
    features = extract_features(normalized_image, use_lbp)

    # Perform clustering
    labels, kmeans = perform_clustering(features, n_clusters, max_iter, init_method)

    # Visualize clustering result
    fig = visualize_clustering(image, labels, n_clusters)
    st.pyplot(fig)

    # Calculate silhouette score
    score = calculate_silhouette_score(features, labels)
    st.write(f"Silhouette Score: {score:.2f}")
    st.write("Silhouette Score explanation: A score closer to 1 indicates better-defined clusters. "
             "Scores above 0.5 generally indicate reasonable to strong clustering structure.")

    # Compare with ground truth if provided
    ground_truth = st.file_uploader("Upload ground truth (optional)", type=["txt", "csv"])
    if ground_truth is not None:
        try:
            gt_labels = np.loadtxt(ground_truth, delimiter=',')
            ari_score = adjusted_rand_score(gt_labels, labels)
            st.write(f"Adjusted Rand Index (comparison with ground truth): {ari_score:.2f}")
            st.write("ARI Score explanation: Ranges from -1 to 1, where 1 indicates perfect agreement "
                     "between the clustering and the ground truth, and values around 0 indicate random labeling.")
        except Exception as e:
            st.error(f"Error loading ground truth file: {e}")
else:
    st.warning("Please select or upload an image before proceeding.")