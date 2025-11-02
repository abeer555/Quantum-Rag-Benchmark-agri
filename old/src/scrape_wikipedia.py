import wikipediaapi
import re
import time
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Expanded list of crops, diseases, and pests (approx. 10x expansion)
WIKI_PAGES = [
    # Major Staple Crops & Grains
    "Wheat", "Rice", "Maize", "Potato", "Soybean", "Barley", "Sorghum",
    "Millet", "Cassava", "Sweet potato", "Oats", "Rye", "Quinoa", "Buckwheat",
    "Teff", "Fonio", "Amaranth", "Spelt", "Kamut", "Triticale", "Freekeh",
    "Emmer", "Einkorn wheat", "Yam", "Taro", "Plantain", "Breadfruit", "Sago",

    # Cash, Oil & Fiber Crops
    "Cotton", "Sugarcane", "Groundnut", "Sunflower", "Rapeseed", "Coconut",
    "Mustard plant", "Oil palm", "Flax", "Jute", "Hemp", "Kenaf", "Ramie",
    "Sisal", "Abacá", "Safflower", "Sesame", "Castor oil plant", "Jojoba",
    "Poppy",

    # Stimulant & Spice Crops
    "Coffee", "Tea plant", "Cacao tree", "Tobacco", "Kola nut", "Yerba mate",
    "Black pepper", "Vanilla", "Clove", "Nutmeg", "Cinnamon", "Cardamom",
    "Ginger", "Turmeric", "Chili pepper", "Allspice", "Anise",

    # Pulses / Legumes
    "Lentil", "Chickpea", "Pigeon pea", "Kidney bean", "Mung bean", "Cowpea",
    "Fava bean", "Lima bean", "Black-eyed pea", "Adzuki bean", "Broad bean",
    "Green pea", "Snow pea", "Sugar snap pea", "Alfalfa", "Clover", "Vetch", "Lupin",

    # Vegetables
    "Tomato", "Onion", "Cabbage", "Spinach", "Eggplant", "Carrot",
    "Cucumber", "Pumpkin", "Okra", "Lettuce", "Kale", "Arugula",
    "Swiss chard", "Collard greens", "Mustard greens", "Bok choy", "Broccoli",
    "Cauliflower", "Brussels sprout", "Radish", "Turnip", "Kohlrabi",
    "Garlic", "Leek", "Shallot", "Chives", "Bell pepper", "Tomatillo",
    "Zucchini", "Gourd", "Watermelon", "Cantaloupe", "Honeydew melon",
    "Parsnip", "Beetroot", "Celery root", "Asparagus", "Artichoke",
    "Celery", "Rhubarb", "Sweet corn",

    # Fruits & Nuts
    "Banana", "Mango", "Apple", "Grapes", "Orange", "Papaya",
    "Pineapple", "Guava", "Pomegranate", "Strawberry", "Avocado",
    "Lemon", "Lime", "Grapefruit", "Pomelo", "Mandarin orange",
    "Peach", "Plum", "Cherry", "Apricot", "Nectarine",
    "Blueberry", "Raspberry", "Blackberry", "Cranberry",
    "Durian", "Lychee", "Jackfruit", "Rambutan", "Dragon fruit",
    "Star fruit", "Passion fruit", "Kiwifruit", "Pear", "Quince",
    "Almond", "Walnut", "Cashew", "Pistachio", "Pecan", "Hazelnut", "Macadamia",
    "Fig", "Date palm",

    # Fungal Diseases
    "Wheat rust", "Rice blast", "Potato late blight", "Coffee leaf rust",
    "Downy mildew", "Powdery mildew", "Anthracnose", "Black Sigatoka",
    "Fusarium wilt", "Verticillium wilt", "Ergot", "Apple scab",
    "Botrytis cinerea", "Clubroot", "Dutch elm disease", "Fire blight",
    "Smut (fungus)", "Southern corn leaf blight", "Sudden oak death",
    "Tar spot of corn", "Wheat leaf rust", "Wheat stem rust", "Wheat stripe rust",
    "Sclerotinia sclerotiorum", "Monilinia fructicola", "Armillaria root rot",

    # Bacterial Diseases
    "Bacterial wilt", "Crown gall", "Citrus canker", "Soft rot",
    "Wildfire (tobacco disease)", "Bacterial leaf blight of rice",
    "Xanthomonas", "Pseudomonas syringae", "Erwinia amylovora",

    # Viral & Viroid Diseases
    "Maize streak virus", "Tomato yellow leaf curl virus",
    "Banana bunchy top virus", "Citrus greening disease",
    "Sugarcane mosaic virus", "Tobacco mosaic virus", "Cucumber mosaic virus",
    "Plum pox", "Barley yellow dwarf", "Cassava mosaic virus",
    "Tomato spotted wilt virus", "Potato spindle tuber viroid",
    "Cadang-cadang",

    # Oomycete Diseases
    "Phytophthora", "Pythium", "Phytophthora cinnamomi",
    "Phytophthora infestans", "Phytophthora ramorum", "Aphanomyces",

    # Nematode Pests
    "Root-knot nematode", "Soybean cyst nematode", "Golden nematode",
    "Burrowing nematode", "Ditylenchus dipsaci",

    # Insect Pests (Lepidoptera - Moths & Butterflies)
    "Fall armyworm", "Cotton bollworm", "Diamondback moth", "Cabbage looper",
    "Codling moth", "European corn borer", "Gypsy moth", "Tobacco hornworm",
    "Helicoverpa zea", "Spodoptera frugiperda", "Plutella xylostella",

    # Insect Pests (Coleoptera - Beetles)
    "Colorado potato beetle", "Japanese beetle", "Boll weevil",
    "Emerald ash borer", "Asian long-horned beetle", "Khapra beetle",
    "Rice weevil", "Western corn rootworm", "Mountain pine beetle",

    # Insect Pests (Hemiptera - True Bugs)
    "Locust", "Whitefly", "Aphid", "Thrips", "Stink bug", "Mealybug",
    "Scale insect", "Glassy-winged sharpshooter", "Tarnished plant bug",
    "Cicada", "Leafhopper", "Psyllid",

    # Other Pests & Weeds
    "Fruit fly", "Spider mite", "Termite", "Earwig", "Grasshopper",
    "Striga", "Cuscuta", "Orobanche", "Parthenium hysterophorus"
]

def clean_text(text: str) -> str:
    """Remove reference markers and normalize spaces."""
    if not text:
        return ""
    text = re.sub(r"\[[0-9]+\]", "", text)  # remove [1], [2], etc.
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fetch_wikipedia_pages(pages=WIKI_PAGES, out_file="wikipedia_agriculture_expanded.txt"):
    """
    Fetch multiple Wikipedia pages and save cleaned text.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="agri_qrag/1.0 (sumit@example.com)"
    )

    total_words = 0
    success = 0
    with open(OUTPUT_DIR / out_file, "w", encoding="utf-8") as fout:
        for p in pages:
            page = wiki.page(p)
            if page.exists():
                text = clean_text(page.text)
                word_count = len(text.split())
                total_words += word_count
                success += 1
                fout.write(f"## {p}\n{text}\n\n")
                print(f"📖 {p}: {word_count} words")
                time.sleep(1)  # polite delay
            else:
                print(f"⚠️ Page not found: {p}")

    print(f"\n✅ Saved {success} pages → {out_file}")
    print(f"📊 Total words collected: {total_words:,}")
    return OUTPUT_DIR / out_file

if __name__ == "__main__":
    fetch_wikipedia_pages()