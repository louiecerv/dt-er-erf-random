#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    text = """Decision Tree, Random Forest and Extreme Random Forest on Random Data Clusters"""
    st.subheader(text)
    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)
 
    st.write('Decision Tree:')
    text = """A very fast classfier but vulnerable to overfitting. May struggle with 
    overlapping clusters due to rigid decision boundaries. Misclassification is 
    likely at the cluster overlap regions.  Simple to interpret, efficient training."""
    st.write(text)
    
    st.write('Random Forest')
    text = """Generally handles overlapping clusters better than decision trees due 
    to averaging predictions from multiple trees. Can still have issues with 
    highly overlapped clusters. Ensemble method, improves robustness and reduces 
    overfitting compared to single decision trees."""
    st.write(text)

    st.write('Extreme Random Forest')
    st.write("""Often shows better performance on overlapping clusters than both 
    decision trees and random forests. This is due to additional randomization in 
    feature selection and splitting criteria. Builds on random forests by 
    introducing additional randomness in feature selection and splitting criteria, 
    potentially improving performance on complex data.""")

    # Create a slider with a label and initial value
    n_samples = st.slider(
        label="Number of samples (200 to 4000):",
        min_value=200,
        max_value=4000,
        step=200,
        value=1000,  # Initial value
    )

    cluster_std = st.number_input("Standard deviation (between 0 and 1):")

    random_state = st.slider(
        label="Random seed (between 0 and 100):",
        min_value=0,
        max_value=100,
        value=42,  # Initial value
    )
   
    n_clusters = st.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )

    # Create the selecton of classifier
    clf = tree.DecisionTreeClassifier()
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)        
    else:
        clf = tree.DecisionTreeClassifier()
        
    if st.button('Start'):
        centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                    cluster_std=cluster_std, centers = centers,
                    random_state=random_state)       
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)
        
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
        st.subheader('VIsualization')
        visualize_classifier(clf, X_test, y_test_pred)
        st.session_state['new_cluster'] = False

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    st.pyplot(fig)

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points

#run the app
if __name__ == "__main__":
    app()
