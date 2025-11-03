"""
Enhanced Agricultural Data Scraper using Crawl4AI
Scrapes comprehensive agricultural data from trusted sources worldwide
Converts all content to clean TXT format for RAG systems
"""

import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking

# Comprehensive agricultural data sources from trusted organizations
# EXPANDED DATASET for better complexity analysis
AGRICULTURAL_SOURCES = {
    # International Organizations - FAO (Expanded)
    "fao_home": "https://www.fao.org/home/en",
    "fao_statistics": "https://www.fao.org/statistics/en/",
    "fao_news": "https://www.fao.org/news/en/",
    "fao_crops": "https://www.fao.org/crop-production/en/",
    "fao_livestock": "https://www.fao.org/animal-production/en/",
    "fao_climate": "https://www.fao.org/climate-smart-agriculture/en/",
    "fao_soils": "https://www.fao.org/soils-portal/en/",
    "fao_water": "https://www.fao.org/land-water/en/",
    "fao_fisheries": "https://www.fao.org/fishery/en/",
    "fao_forestry": "https://www.fao.org/forestry/en/",
    "fao_family_farming": "https://www.fao.org/family-farming/en/",
    "fao_food_security": "https://www.fao.org/food-security/en/",
    "fao_nutrition": "https://www.fao.org/nutrition/en/",
    "fao_gender": "https://www.fao.org/gender/en/",
    "fao_rural_development": "https://www.fao.org/rural-development/en/",
    
    # US Government - USDA (Expanded)
    "usda_farming": "https://www.usda.gov/topics/farming",
    "usda_crops": "https://www.usda.gov/topics/crops",
    "usda_livestock": "https://www.usda.gov/topics/livestock",
    "usda_research": "https://www.usda.gov/topics/research-and-science",
    "usda_conservation": "https://www.usda.gov/topics/conservation",
    "usda_organic": "https://www.usda.gov/topics/organic",
    "usda_ers": "https://www.ers.usda.gov/",
    "usda_nass": "https://www.nass.usda.gov/",
    "usda_biotechnology": "https://www.usda.gov/topics/biotechnology",
    "usda_food_nutrition": "https://www.usda.gov/topics/food-and-nutrition",
    "usda_trade": "https://www.usda.gov/topics/trade",
    "usda_rural": "https://www.usda.gov/topics/rural",
    "usda_forestry": "https://www.usda.gov/topics/forestry",
    "usda_climate": "https://www.usda.gov/topics/climate-solutions",
    
    # World Bank (Expanded)
    "worldbank_agriculture": "https://www.worldbank.org/en/topic/agriculture",
    "worldbank_food_security": "https://www.worldbank.org/en/topic/agriculture/brief/food-security-and-covid-19",
    "worldbank_climate_smart": "https://www.worldbank.org/en/topic/climate-smart-agriculture",
    "worldbank_rural_dev": "https://www.worldbank.org/en/topic/rural-development",
    
    # Research Institutions - CGIAR Centers (Expanded)
    "cgiar": "https://www.cgiar.org/",
    "cgiar_research": "https://www.cgiar.org/research/",
    "ifpri": "https://www.ifpri.org/",
    "cimmyt": "https://www.cimmyt.org/",
    "cimmyt_wheat": "https://www.cimmyt.org/work/wheat/",
    "cimmyt_maize": "https://www.cimmyt.org/work/maize/",
    "irri": "https://www.irri.org/",
    "irri_rice_knowledge": "https://www.irri.org/rice-knowledge",
    "icrisat": "https://www.icrisat.org/",
    "icrisat_dryland": "https://www.icrisat.org/dryland-cereals/",
    "worldfish": "https://www.worldfishcenter.org/",
    "ilri": "https://www.ilri.org/",
    "cifor": "https://www.cifor-icraf.org/",
    "bioversity": "https://www.bioversityinternational.org/",
    
    # UK Government (Expanded)
    "defra": "https://www.gov.uk/government/organisations/department-for-environment-food-rural-affairs",
    "defra_farming": "https://www.gov.uk/government/policies/farming",
    "defra_food_farming": "https://www.gov.uk/topic/farming-food-grants-payments",
    "uk_agriculture_horticulture": "https://www.gov.uk/topic/environmental-management/agriculture-horticulture",
    
    # EU Agriculture (Expanded)
    "eu_agriculture": "https://agriculture.ec.europa.eu/",
    "eu_cap": "https://agriculture.ec.europa.eu/common-agricultural-policy_en",
    "eu_farming": "https://agriculture.ec.europa.eu/farming_en",
    "eu_rural_development": "https://agriculture.ec.europa.eu/cap-my-country/cap-strategic-plans_en",
    "eu_organic": "https://agriculture.ec.europa.eu/farming/organic-farming_en",
    
    # Australia (Expanded)
    "australia_agriculture": "https://www.agriculture.gov.au/",
    "csiro_agriculture": "https://www.csiro.au/en/research/animals-and-agriculture",
    "australia_crops": "https://www.agriculture.gov.au/agriculture-land/farm-food-drought/crops",
    "australia_livestock": "https://www.agriculture.gov.au/agriculture-land/farm-food-drought/livestock",
    
    # Canada
    "canada_agriculture": "https://agriculture.canada.ca/en",
    "canada_crops": "https://agriculture.canada.ca/en/sector/crops",
    "canada_livestock": "https://agriculture.canada.ca/en/sector/livestock",
    
    # India (Expanded)
    "icar": "https://icar.org.in/",
    "india_agriculture_ministry": "https://agricoop.nic.in/",
    
    # China
    "china_agriculture": "http://www.moa.gov.cn/",
    
    # Brazil
    "brazil_embrapa": "https://www.embrapa.br/en/home",
    
    # Agricultural Science Journals & Databases
    "agris_fao": "https://agris.fao.org/",
    "nature_food": "https://www.nature.com/nfood/",
    
    # Climate & Agriculture (Expanded)
    "ipcc_agriculture": "https://www.ipcc.ch/",
    "climate_agriculture_alliance": "https://www.wri.org/climate/agriculture",
    
    # Sustainable Agriculture (Expanded)
    "sustainable_ag_research": "https://www.sustainableagriculture.net/",
    "regenerative_agriculture": "https://regenerationinternational.org/",
    "organic_farming": "https://www.ifoam.bio/",
    
    # Precision Agriculture (Expanded)
    "precision_ag": "https://www.precisionag.com/",
    "farm_technology": "https://www.agriculture.com/farm-management/technology",
    
    # Agricultural Technology & Innovation
    "agfunder": "https://agfundernews.com/",
    "agtech_news": "https://www.agtechinnovator.com/",
    "future_farming": "https://www.futurefarming.com/",
    
    # Soil Science
    "soil_science_society": "https://www.soils.org/",
    "global_soil_partnership": "https://www.fao.org/global-soil-partnership/en/",
    
    # Water Management
    "water_agriculture": "https://www.iwmi.cgiar.org/",
    "irrigation_drainage": "https://www.icid.org/",
    
    # Seed & Genetics
    "crop_trust": "https://www.croptrust.org/",
    "seed_systems": "https://www.cabi.org/what-we-do/agriculture-and-biosecurity/seed-systems/",
    
    # Agricultural Economics
    "agricultural_economics": "https://www.choicesmagazine.org/",
    "farm_economics": "https://farmdocdaily.illinois.edu/",
    
    # Pest Management
    "ipm_integrated": "https://www.ipmcenters.org/",
    "pesticide_info": "https://npic.orst.edu/",
    
    # Livestock & Animal Science
    "livestock_research": "https://www.ilri.org/research",
    "animal_health": "https://www.woah.org/",
    
    # Agricultural Extension
    "extension_agriculture": "https://www.extension.org/",
    
    # Food Systems
    "food_systems": "https://www.un.org/en/food-systems-summit",
    
    # Agroforestry
    "agroforestry": "https://www.worldagroforestry.org/",
}

class AgricultureDataScraper:
    def __init__(self, output_dir="agricultural_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.output_dir / "txt").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_scraped": 0,
            "successful": 0,
            "failed": 0,
            "total_chars": 0,
            "sources": []
        }
    
    def clean_text(self, text):
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\n]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def markdown_to_txt(self, markdown_text):
        """Convert markdown to plain text"""
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', markdown_text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove markdown bold/italic **text** or *text* -> text
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        
        # Remove markdown code blocks
        text = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove markdown images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        return text
    
    async def scrape_to_txt(self, url, name, max_retries=3):
        """Scrape content and save as clean TXT file"""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=True,
            page_timeout=60000,  # 60 seconds timeout
        )
        
        for attempt in range(max_retries):
            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config
                    )
                    
                    if result.success:
                        # Convert markdown to plain text
                        plain_text = self.markdown_to_txt(result.markdown)
                        clean_text = self.clean_text(plain_text)
                        
                        # Create header
                        header = f"""{'='*80}
SOURCE: {name}
URL: {url}
SCRAPED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
CONTENT LENGTH: {len(clean_text)} characters
{'='*80}

"""
                        
                        final_content = header + clean_text
                        
                        # Save as TXT
                        txt_file = self.output_dir / "txt" / f"{name}.txt"
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(final_content)
                        
                        # Save metadata as JSON
                        metadata = {
                            "source_name": name,
                            "url": url,
                            "scraped_at": datetime.now().isoformat(),
                            "content_length": len(clean_text),
                            "success": True,
                            "links_count": len(result.links) if result.links else 0,
                            "txt_file": str(txt_file)
                        }
                        
                        json_file = self.output_dir / "json" / f"{name}_metadata.json"
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Update stats
                        self.stats["successful"] += 1
                        self.stats["total_chars"] += len(clean_text)
                        self.stats["sources"].append(name)
                        
                        print(f"✓ [{self.stats['successful']:3d}] {name:40s} | {len(clean_text):8d} chars")
                        return result
                    else:
                        if attempt < max_retries - 1:
                            print(f"⚠ Retry {attempt + 1}/{max_retries} for {name}")
                            await asyncio.sleep(2)
                        else:
                            print(f"✗ Failed {name}: {result.error_message}")
                            self.stats["failed"] += 1
                            return None
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠ Exception retry {attempt + 1}/{max_retries} for {name}: {str(e)}")
                    await asyncio.sleep(2)
                else:
                    print(f"✗ Exception in {name}: {str(e)}")
                    self.stats["failed"] += 1
                    return None
        
        return None
    
    async def scrape_with_links_extraction(self, url, name):
        """Extract internal links for deeper crawling"""
        browser_config = BrowserConfig(headless=True, verbose=False)
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            exclude_external_links=False
        )
        
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=crawl_config)
                
                if result.success and result.links:
                    # Extract relevant agricultural links
                    agri_keywords = [
                        'crop', 'farm', 'agri', 'livestock', 'soil', 
                        'irrigation', 'harvest', 'production', 'food',
                        'rural', 'research', 'statistics', 'data'
                    ]
                    
                    relevant_links = []
                    all_links = []
                    
                    if isinstance(result.links, dict):
                        all_links = result.links.get("internal", []) + result.links.get("external", [])
                    elif isinstance(result.links, list):
                        all_links = result.links
                    
                    for link_data in all_links:
                        if isinstance(link_data, dict):
                            href = link_data.get("href", "")
                            text = link_data.get("text", "")
                        else:
                            href = str(link_data)
                            text = ""
                        
                        # Check if link is relevant
                        if any(kw in href.lower() or kw in text.lower() for kw in agri_keywords):
                            relevant_links.append({
                                "url": href,
                                "text": text
                            })
                    
                    # Save links
                    links_file = self.output_dir / "json" / f"{name}_links.json"
                    with open(links_file, 'w', encoding='utf-8') as f:
                        json.dump(relevant_links[:100], f, indent=2)  # Save top 100
                    
                    print(f"  → Found {len(relevant_links)} relevant links from {name}")
                    return relevant_links
                    
        except Exception as e:
            print(f"  → Error extracting links from {name}: {str(e)}")
        
        return []
    
    def save_statistics(self):
        """Save scraping statistics"""
        self.stats["total_scraped"] = self.stats["successful"] + self.stats["failed"]
        
        stats_file = self.output_dir / "logs" / f"scraping_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create summary report
        summary = f"""
{'='*80}
AGRICULTURAL DATA SCRAPING SUMMARY
{'='*80}
Total Sources Attempted: {self.stats['total_scraped']}
Successful: {self.stats['successful']}
Failed: {self.stats['failed']}
Success Rate: {(self.stats['successful']/self.stats['total_scraped']*100) if self.stats['total_scraped'] > 0 else 0:.1f}%
Total Content: {self.stats['total_chars']:,} characters ({self.stats['total_chars']/1024/1024:.2f} MB)
Average per Source: {self.stats['total_chars']//self.stats['successful'] if self.stats['successful'] > 0 else 0:,} characters

Output Directory: {self.output_dir.absolute()}
  - TXT files: {self.output_dir / 'txt'}
  - Metadata: {self.output_dir / 'json'}
  - Logs: {self.output_dir / 'logs'}
{'='*80}
"""
        
        summary_file = self.output_dir / "logs" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
    
    async def scrape_all_sources(self, sources_dict, delay=2):
        """Scrape all sources with delay between requests"""
        print(f"\n{'='*80}")
        print(f"Starting to scrape {len(sources_dict)} agricultural sources...")
        print(f"{'='*80}\n")
        
        for name, url in sources_dict.items():
            await self.scrape_to_txt(url, name)
            # Be polite - add delay between requests
            await asyncio.sleep(delay)
        
        # Save final statistics
        self.save_statistics()
    
    def merge_all_txt_files(self, output_filename="agricultural_corpus_complete.txt"):
        """Merge all scraped TXT files into one large corpus"""
        txt_dir = self.output_dir / "txt"
        all_txt_files = sorted(txt_dir.glob("*.txt"))
        
        merged_file = self.output_dir / output_filename
        
        with open(merged_file, 'w', encoding='utf-8') as outfile:
            outfile.write(f"""{'='*80}
COMPREHENSIVE AGRICULTURAL CORPUS
Compiled: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Sources: {len(all_txt_files)}
{'='*80}

""")
            
            for txt_file in all_txt_files:
                print(f"Merging: {txt_file.name}")
                with open(txt_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n" + "="*80 + "\n\n")
        
        file_size = merged_file.stat().st_size / (1024 * 1024)  # MB
        print(f"\n✓ Created merged corpus: {merged_file}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Sources: {len(all_txt_files)}")
        
        return merged_file



async def main():
    """Main scraping function - scrapes all trusted agricultural sources"""
    scraper = AgricultureDataScraper(output_dir="agricultural_data_complete")
    
    print(f"""
{'='*80}
COMPREHENSIVE AGRICULTURAL DATA SCRAPER
{'='*80}
This scraper will collect data from {len(AGRICULTURAL_SOURCES)} trusted sources:
  • FAO (Food and Agriculture Organization)
  • USDA (US Department of Agriculture)
  • World Bank Agriculture
  • CGIAR Research Centers
  • Government Agricultural Departments (UK, EU, Australia, India)
  • Research Institutions and Journals
  
All content will be converted to clean TXT format for RAG systems.
{'='*80}
""")
    
    # Scrape all sources
    await scraper.scrape_all_sources(AGRICULTURAL_SOURCES, delay=3)
    
    # Merge all files into one corpus
    print(f"\n{'='*80}")
    print("Creating unified agricultural corpus...")
    print(f"{'='*80}\n")
    scraper.merge_all_txt_files()
    
    print(f"\n{'='*80}")
    print("✓ SCRAPING COMPLETE!")
    print(f"{'='*80}")


async def scrape_specific_topics():
    """Scrape specific agricultural topics with expanded sources"""
    scraper = AgricultureDataScraper(output_dir="agricultural_topics_expanded")
    
    # Expanded topic-specific sources
    topics = {
        # Crop Production & Management
        "fao_crops_production": "https://www.fao.org/crop-production/en/",
        "fao_plant_production": "https://www.fao.org/plant-production-protection/en/",
        "usda_crop_production": "https://www.usda.gov/topics/crops",
        "fao_seeds": "https://www.fao.org/seeds/en/",
        
        # Livestock & Animal Production
        "fao_livestock_production": "https://www.fao.org/animal-production/en/",
        "fao_animal_health": "https://www.fao.org/animal-health/en/",
        "usda_livestock_production": "https://www.usda.gov/topics/livestock",
        
        # Climate Smart Agriculture
        "fao_climate_smart": "https://www.fao.org/climate-smart-agriculture/en/",
        "fao_climate_change": "https://www.fao.org/climate-change/en/",
        
        # Soil & Water Management
        "fao_soils_management": "https://www.fao.org/soils-portal/en/",
        "fao_water_management": "https://www.fao.org/land-water/en/",
        "fao_irrigation": "https://www.fao.org/land-water/water/en/",
        
        # Agricultural Technology & Innovation
        "fao_innovation": "https://www.fao.org/innovation/en/",
        "fao_digital_agriculture": "https://www.fao.org/digital-agriculture/en/",
        
        # Food Security & Nutrition
        "fao_food_security": "https://www.fao.org/food-security/en/",
        "fao_nutrition": "https://www.fao.org/nutrition/en/",
        
        # Sustainable Agriculture
        "fao_sustainable_agriculture": "https://www.fao.org/sustainability/en/",
        "fao_organic_agriculture": "https://www.fao.org/organicag/en/",
        "fao_agroecology": "https://www.fao.org/agroecology/en/",
        
        # Agricultural Economics & Trade
        "fao_markets_trade": "https://www.fao.org/markets-and-trade/en/",
        "fao_agribusiness": "https://www.fao.org/ag/ags/en/",
        
        # Forestry & Agroforestry
        "fao_forestry_agriculture": "https://www.fao.org/forestry/en/",
        
        # Fisheries & Aquaculture
        "fao_fisheries_aquaculture": "https://www.fao.org/fishery/en/",
        
        # Research & Statistics
        "fao_agri_statistics": "https://www.fao.org/statistics/en/",
        "usda_agricultural_research": "https://www.usda.gov/topics/research-and-science",
        "usda_statistics": "https://www.nass.usda.gov/",
    }
    
    print(f"Scraping {len(topics)} topic-specific sources...")
    await scraper.scrape_all_sources(topics, delay=3)
    
    # Merge all topic files
    scraper.merge_all_txt_files("agricultural_topics_corpus.txt")


async def scrape_research_institutions():
    """Scrape from major agricultural research institutions"""
    scraper = AgricultureDataScraper(output_dir="agricultural_research")
    
    research_sources = {
        # CGIAR Centers
        "cgiar_home": "https://www.cgiar.org/",
        "cgiar_food_security": "https://www.cgiar.org/research/food-land-and-water-systems/",
        "cimmyt_maize_wheat": "https://www.cimmyt.org/",
        "irri_rice": "https://www.irri.org/",
        "icrisat_dryland": "https://www.icrisat.org/",
        "cip_potato": "https://cipotato.org/",
        "iita_africa": "https://www.iita.org/",
        "icarda_dryland": "https://www.icarda.org/",
        "worldfish": "https://www.worldfishcenter.org/",
        
        # Other Research Institutions
        "ifpri_policy": "https://www.ifpri.org/",
        "cabi_agriculture": "https://www.cabi.org/",
        
        # National Research
        "csiro_australia": "https://www.csiro.au/en/research/animals-and-agriculture",
        "rothamsted_uk": "https://www.rothamsted.ac.uk/",
        "icar_india": "https://icar.org.in/",
    }
    
    print(f"Scraping {len(research_sources)} research institutions...")
    await scraper.scrape_all_sources(research_sources, delay=3)
    
    scraper.merge_all_txt_files("agricultural_research_corpus.txt")


async def scrape_quick_sample():
    """Quick sample scraping for testing (10 sources)"""
    scraper = AgricultureDataScraper(output_dir="agricultural_sample")
    
    sample_sources = {
        "fao_home": "https://www.fao.org/home/en",
        "fao_crops": "https://www.fao.org/crop-production/en/",
        "usda_farming": "https://www.usda.gov/topics/farming",
        "worldbank_agriculture": "https://www.worldbank.org/en/topic/agriculture",
        "cgiar_research": "https://www.cgiar.org/research/",
        "fao_climate": "https://www.fao.org/climate-smart-agriculture/en/",
        "usda_organic": "https://www.usda.gov/topics/organic",
        "fao_soils": "https://www.fao.org/soils-portal/en/",
        "ifpri": "https://www.ifpri.org/",
        "irri": "https://www.irri.org/",
    }
    
    print(f"Quick sample: Scraping {len(sample_sources)} sources for testing...")
    await scraper.scrape_all_sources(sample_sources, delay=2)
    
    scraper.merge_all_txt_files("agricultural_sample_corpus.txt")


if __name__ == "__main__":
    """
    Installation:
    pip install crawl4ai
    
    Usage:
    1. Full scraping (50+ sources): python web_crawler.py
    2. Quick test (10 sources): uncomment scrape_quick_sample()
    3. Topics only: uncomment scrape_specific_topics()
    4. Research institutions: uncomment scrape_research_institutions()
    """
    
    # Choose one of the following:
    
    # 1. FULL COMPREHENSIVE SCRAPING (Recommended)
    asyncio.run(main())
    
    # 2. Quick sample for testing (uncomment to use)
    # asyncio.run(scrape_quick_sample())
    
    # 3. Topic-specific scraping (uncomment to use)
    # asyncio.run(scrape_specific_topics())
    
    # 4. Research institutions only (uncomment to use)
    # asyncio.run(scrape_research_institutions())
