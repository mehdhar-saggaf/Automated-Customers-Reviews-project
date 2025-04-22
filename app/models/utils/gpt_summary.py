# utils/gpt_summary.py
import openai
import os

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4-1106-preview"

def generate_category_title(meta_category, product_names):
    prompt = (
        f"You are an expert marketer. Below are product names from a category labeled '{meta_category}':\n"
        f"{', '.join(product_names[:10])}\n"
        "Provide a short, creative, and descriptive title for this product category."
    )
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def summarize_category(meta_category, products_info, full_df):
    import pandas as pd

    # Filter relevant data
    df = full_df[full_df["meta_category"] == meta_category]
    df = df.dropna(subset=["name", "reviews.rating", "reviews.text"])

    # Top 3 highest-rated products
    avg_ratings = df.groupby("name")["reviews.rating"].mean()
    top_products = avg_ratings.sort_values(ascending=False).head(3)
    top_list = "\n".join(
        [f"{i+1}. {name} (Avg Rating: {avg_ratings[name]:.2f})" for i, name in enumerate(top_products.index)]
    )

    # Worst-rated product
    worst_product = avg_ratings.sort_values().head(1)
    if worst_product.empty:
        worst = "Unknown"
        joined_worst_reviews = "No reviews available."
    else:
        worst = worst_product.index[0]
        worst_reviews = df[df["name"] == worst]["reviews.text"].astype(str).tolist()
        joined_worst_reviews = "\n".join(worst_reviews[:10])

    # Format input reviews
    limited_info = [info[:400] for info in products_info[:40]]
    formatted_reviews = "\n".join(limited_info)

    # Prompt
    prompt = f"""
You are a senior product analyst with expertise in interpreting customer behavior across varied contexts.
Given customer reviews for the category [{meta_category}], write a well-structured blog-style article that includes the following:
---
1. **Top 3 Highest-Rated Products**
   - Based on the ratings, the top 3 products are:
{top_list}
   - Use the real product names listed above in your analysis.
   - Summarize their key strengths in bullet points.
   - Highlight unique features or differences between them.
   - If applicable, explain how user context (e.g., frequent travelers, gamers, remote workers) influenced their positive experiences.

2. **Most Common Complaints per Product**
   - For each top product, summarize the most frequent issues or complaints mentioned.
   - Pay close attention to issues related to **environmental or contextual conditions** (e.g., hot or cold weather, indoor vs outdoor use, long vs short usage time), **explicitly mention that context**.

3. **Worst-Rated Product**
   - Product: {worst}
   - Reviews: {joined_worst_reviews}
   - Clearly summarize what made the user experience negative.
   - If the complaints seem to arise in specific usage conditions, mention them.

4. **Final Recommendation**
   - Recommend the best product from the top 3.
   - Justify your recommendation using comparative reasoning based on the data.
   - Optionally suggest what type of user (e.g., budget-focused, frequent traveler, home user) would benefit most.
---
**Style & Output Instructions:**
- Keep the tone professional and objective, make it engaging for the reader.
- Use clear subheadings to structure the article.
Begin your analysis now.
"""

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()
