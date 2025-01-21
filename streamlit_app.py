import streamlit as st
import pandas as pd
import warnings
import sys
import os
import pickle
from fuzzywuzzy import process


# Get the absolute path to the current directory and data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, 'data', 'foods_data.csv')

# Add src path to sys
sys.path.append(os.path.abspath('src'))
from remove_ import remove

# Suppress specific future warnings from the transformers module
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Suppress DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set up the page configuration
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv(data_file)

df1 = pickle.load(file=open(file=r'src/model/fooddataframe.pkl', mode='rb'))
similarity = pickle.load(file=open(file=r'src/model/foodsimilarity.pkl', mode='rb'))

remove()

def main():
    
    # st.title('Mobile Recommendation System')
    st.sidebar.title('Navigation')

    # Sidebar navigation
    options = st.sidebar.radio('Select an Option', ['Home', 'Variety Recommendations','Search Recommendations'])
    
    if options == 'Home':
        show_home()
    elif options == 'Variety Recommendations':
        show_recommendations()
    elif options == 'Search Recommendations':
        show_search_recommendations()
            


def show_home():
    # Setting the page background color
    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Adding a hero section
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Welcome to Sysco Food Store</h1>", unsafe_allow_html=True)
    
    # Adding a catchy tagline
    st.markdown("<h3 style='text-align: center; color: #1c6b8c;'>Your one-stop shop for the best food items!</h3>", unsafe_allow_html=True)
    
    # Displaying a welcoming image
    st.image('sysco.webp', use_container_width=True, caption='Find your preffered food items')

    # Adding a section with a call to action
    st.markdown("<p style='text-align: center; font-size: 18px;'>Let us guide you through our vast selection of high-quality food products tailored to your needs. Whether you’re looking for premium ingredients or budget-friendly essentials, we've got you covered!</p>", unsafe_allow_html=True)
    
    # Adding some space
    st.write("\n\n")
    
    # Highlighting categories or offers
    st.subheader("Best Food Recommendation App")

def show_recommendations():
    st.markdown("""
        <style>
        .food-image {
            max-width: 150px;
            max-height: 200px;
            object-fit: cover;
            margin-bottom: 10px;
        }
        .recommendation-item {
            border: 1px solid #e0e0e0;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header('Food Recommendations')

    # Filter by Name
    food_item = st.selectbox('Select a Food Item', df['name'].unique())

    # Add a button to trigger the recommendation display
    if st.button('Show Recommendations'):
        # Get recommendations from the recommend function
        recommended_foods, recommended_foods_IMG, recommended_foods_ratings, recommended_foods_no_of_ratings, \
            recommended_foods_discount_price, recommended_foods_acount_price = recommend(food_item)

        # Display Recommendations as a Grid
        num_cols = 4  # Number of columns in the grid
        num_rows = (len(recommended_foods) + num_cols - 1) // num_cols  # Calculate rows based on the number of items

        for row_index in range(num_rows):
            cols = st.columns(num_cols)  # Create columns for the current row
            for col_index in range(num_cols):
                item_index = row_index * num_cols + col_index
                if item_index < len(recommended_foods):
                    # Fetch data for the current item
                    food_name = recommended_foods[item_index]
                    food_img = recommended_foods_IMG[item_index]
                    food_rating = recommended_foods_ratings[item_index]
                    food_no_of_ratings = recommended_foods_no_of_ratings[item_index]
                    food_discount_price = recommended_foods_discount_price[item_index]
                    food_actual_price = recommended_foods_acount_price[item_index]

                    with cols[col_index]:
                        # Display item details in the column
                        st.markdown(f'<div class="recommendation-item">'
                                    f'<img src="{food_img}" class="food-image">'
                                    f'<h3>{food_name}</h3>'
                                    f'<p>Ratings: {food_rating}</p>'
                                    f'<p>No of Ratings: {food_no_of_ratings}</p>'
                                    f'<p>Discount Price: USD {food_discount_price}</p>'
                                    f'<p>Actual Price: USD {food_actual_price}</p>'
                                    f'</div>', unsafe_allow_html=True)

        # If there are no recommendations found
        if not recommended_foods:
            st.write('No similar food items found.')

def show_search_recommendations():
    st.markdown("""
        <style>
        .food-image {
            max-width: 150px;
            max-height: 200px;
            object-fit: cover;
            margin-bottom: 10px;
        }
        .recommendation-item {
            border: 1px solid #e0e0e0;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header('Food Recommendations by Search')

    # Add a search input bar for typing food item names
    input_item = st.text_input('Search for a Food Item')

    # Add a button to trigger the recommendation display
    if st.button('Search and Show Recommendations'):
        # Get recommendations based on search input
        recommendations = search_recommendations(input_item, df)

        if not recommendations.empty:
            # Extract data from the recommendations DataFrame
            recommended_foods = recommendations['name'].tolist()
            recommended_foods_IMG = recommendations['image'].tolist()
            recommended_foods_ratings = recommendations['ratings'].tolist()
            recommended_foods_no_of_ratings = recommendations['no_of_ratings'].tolist()
            recommended_foods_discount_price = recommendations['discount_price'].fillna('Not Available').tolist()
            recommended_foods_actual_price = recommendations['actual_price'].tolist()

            # Display Recommendations as a Grid
            num_cols = 4  # Number of columns in the grid
            num_rows = (len(recommended_foods) + num_cols - 1) // num_cols  # Calculate rows based on the number of items

            for row_index in range(num_rows):
                cols = st.columns(num_cols)  # Create columns for the current row
                for col_index in range(num_cols):
                    item_index = row_index * num_cols + col_index
                    if item_index < len(recommended_foods):
                        # Fetch data for the current item
                        food_name = recommended_foods[item_index]
                        food_img = recommended_foods_IMG[item_index]
                        food_rating = recommended_foods_ratings[item_index]
                        food_no_of_ratings = recommended_foods_no_of_ratings[item_index]
                        food_discount_price = recommended_foods_discount_price[item_index]
                        food_actual_price = recommended_foods_actual_price[item_index]

                        with cols[col_index]:
                            # Display item details in the column
                            st.markdown(f'<div class="recommendation-item">'
                                        f'<img src="{food_img}" class="food-image">'
                                        f'<h3>{food_name}</h3>'
                                        f'<p>Ratings: {food_rating}</p>'
                                        f'<p>No of Ratings: {food_no_of_ratings}</p>'
                                        f'<p>Discount Price: USD {food_discount_price}</p>'
                                        f'<p>Actual Price: USD {food_actual_price}</p>'
                                        f'</div>', unsafe_allow_html=True)
        else:
            # If there are no recommendations found
            st.write('No matching food items found. Try a different search term.')


def search_recommendations(input_item, df, top_n=5):
    # Define the priority keyword to prioritize certain matches
    priority_keyword = "sysco"  # Hardcoded keyword to prioritize matches

    # Ensure food names are converted to a list of strings
    food_inventory = df["name"].astype(str).tolist()

    # Find matches using fuzzy matching
    matches = process.extract(input_item, food_inventory, limit=top_n)

    if not matches:
        print(f"No matches found for '{input_item}'.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches are found

    # Separate matches into priority and others based on the keyword
    keyword_matches = [match for match in matches if priority_keyword.lower() in match[0].lower()]
    other_matches = [match for match in matches if priority_keyword.lower() not in match[0].lower()]

    # Combine keyword matches first, followed by other matches
    matches = keyword_matches + other_matches

    # Extract matched product names and their similarity scores
    matched_names = [match[0] for match in matches]

    # Filter the DataFrame to include only matched products
    recommendations = df[df["name"].isin(matched_names)].copy()

    # Add a similarity score column to the filtered recommendations
    recommendations["similarity_score"] = recommendations["name"].apply(
        lambda name: next(score for item, score in matches if item == name)
    )

    # Sort recommendations by priority (keyword presence) and similarity score
    recommendations["is_priority"] = recommendations["name"].str.contains(priority_keyword, case=False, na=False)
    recommendations = recommendations.sort_values(
        by=["is_priority", "similarity_score"], ascending=[False, False]
    ).drop(columns=["is_priority"])  # Drop the temporary priority column

    return recommendations


def recommend(food_item):
    # Define the priority keyword to prioritize certain matches
    priority_keyword = "sysco"  # Hardcoded keyword to prioritize matches

    # Ensure food names are converted to a list of strings
    food_inventory = df['name'].astype(str).tolist()

    # Find all matches for the food item in the dataset using fuzzy matching
    results = process.extract(food_item, food_inventory, limit=10)

    if not results:
        print(f"The food item '{food_item}' did not match any item in the database.")
        return [], [], [], [], [], []

    # Sort results by similarity score (descending) and name (alphabetical)
    results = sorted(results, key=lambda x: (-x[1], x[0]))

    # Separate items containing the priority keyword
    keyword_results = [res for res in results if priority_keyword.lower() in res[0].lower()]
    other_results = [res for res in results if priority_keyword.lower() not in res[0].lower()]

    # Combine priority matches and others
    results = keyword_results + other_results

    # Extract matched food names and their similarity scores
    matched_names = [res[0] for res in results]
    similarity_scores = [res[1] for res in results]

    # Find indices for the matched names
    food_indices = [df[df['name'] == name].index[0] for name in matched_names]

    # Initialize lists for recommendations
    recommended_foods = []
    recommended_foods_IMG = []
    recommended_foods_ratings = []
    recommended_foods_no_of_ratings = []
    recommended_foods_discount_price = []
    recommended_foods_acount_price = []

    for idx in food_indices:
        # Append details for each recommended food item
        recommended_foods.append(df['name'].iloc[idx])
        recommended_foods_IMG.append(fetch_IMG(idx))
        recommended_foods_ratings.append(df['ratings'].iloc[idx])
        recommended_foods_no_of_ratings.append(df['no_of_ratings'].iloc[idx])
        recommended_foods_discount_price.append(df['discount_price'].iloc[idx])
        recommended_foods_acount_price.append(df['actual_price'].iloc[idx])

    return (recommended_foods, recommended_foods_IMG, recommended_foods_ratings, 
            recommended_foods_no_of_ratings, recommended_foods_discount_price, 
            recommended_foods_acount_price)

def fetch_IMG(food_index):
    return df['image'].iloc[food_index]

if __name__ == '__main__':
    main()
