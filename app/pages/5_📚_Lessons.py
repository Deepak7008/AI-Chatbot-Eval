import streamlit as st
import os

st.set_page_config(page_title="Lessons & Insights", page_icon="📚", layout="wide")

# Determine path to tasks/lessons.md relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
lessons_path = os.path.join(project_root, "tasks", "lessons.md")

if not os.path.exists(lessons_path):
    st.error(f"Cannot find lessons file at: `{lessons_path}`\nPlease ensure the file exists.")
    st.stop()

with open(lessons_path, "r", encoding="utf-8") as f:
    content = f.read()

# --- Parse the Markdown file ---
# We split by "\n## " to safely catch headers (adding newline prevents catching mid-text ##)
if content.startswith("## "):
    parts = [""] + content[3:].split("\n## ")
else:
    parts = content.split("\n## ")

header_content = parts[0]
sections_content = parts[1:]

# Extract unique categories while preserving their order of appearance
all_categories = []
for section in sections_content:
    lines = section.split("\n", 1)
    title_line = lines[0].strip()
    
    # Extract the category name (everything before the first '/')
    if "/" in title_line:
        category = title_line.split("/")[0].strip()
    else:
        category = title_line.strip()
        
    if category not in all_categories:
        all_categories.append(category)

# --- UI Layout ---
st.title("📚 Lessons & Insights")
st.markdown("Filter and review architectural lessons, bugs, and best practices learned during development.")

# Multiselect filter for categories
selected_categories = st.multiselect(
    "Filter by Category", 
    options=all_categories, 
    default=all_categories,
    help="Select categories to view specific lessons. Categories are automatically derived from the headers in lessons.md."
)

st.divider()

# Render the top section of the markdown file (e.g. the main `# Lessons Learned` title and intro)
if header_content.strip():
    st.markdown(header_content)

# Render the filtered sections while maintaining the original document order
if not selected_categories:
    st.info("No categories selected. Please select at least one category from the dropdown above.")
else:
    filtered_markdown = ""
    for section in sections_content:
        lines = section.split("\n", 1)
        title_line = lines[0].strip()
        
        if "/" in title_line:
            category = title_line.split("/")[0].strip()
        else:
            category = title_line.strip()
            
        if category in selected_categories:
            filtered_markdown += "## " + section + "\n\n"
            
    st.markdown(filtered_markdown)
