"""
Debug script to test Wikipedia retrieval for specific entities/claims
Usage: python debug_wikipedia.py
"""
import pandas as pd
import wikipedia
import time

def test_entity_lookup(entity_name, verbose=True):
    """Test if an entity can be found on Wikipedia"""
    wikipedia.set_lang("en")
    
    print(f"\n🔍 Testing entity: '{entity_name}'")
    print("-" * 60)
    
    # Try direct lookup
    try:
        time.sleep(0.5)
        page = wikipedia.page(entity_name, auto_suggest=True)
        if verbose:
            print(f"✅ Found via direct lookup!")
            print(f"   Title: {page.title}")
            print(f"   URL: {page.url}")
            print(f"   Content length: {len(page.content)} chars")
        return True
    except wikipedia.exceptions.DisambiguationError as e:
        if verbose:
            print(f"⚠️  Disambiguation page with {len(e.options)} options")
            print(f"   Options: {e.options[:5]}")  # Show first 5
        return False
    except wikipedia.exceptions.PageError as e:
        if verbose:
            print(f"❌ Page not found")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Error: {type(e).__name__}: {str(e)[:100]}")
        return False

def test_claim_search(claim, verbose=True):
    """Test if a claim returns search results"""
    wikipedia.set_lang("en")
    
    print(f"\n🔍 Testing claim search: '{claim[:60]}...'")
    print("-" * 60)
    
    try:
        time.sleep(0.5)
        results = wikipedia.search(claim, results=5)
        if verbose:
            if results:
                print(f"✅ Found {len(results)} search results")
                for i, title in enumerate(results, 1):
                    print(f"   {i}. {title}")
            else:
                print(f"❌ No search results found")
        return len(results) > 0
    except Exception as e:
        if verbose:
            print(f"❌ Error: {type(e).__name__}: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    df = pd.read_csv("processed_fakenews_data.csv")
    
    print("=" * 60)
    print("Wikipedia Entity/Claim Debugging Tool")
    print("=" * 60)
    
    # Test first 10 claims
    print(f"\n📊 Testing first 10 claims from your dataset...\n")
    
    success_entities = 0
    success_claims = 0
    
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        claim = str(row['claim'])
        entities = str(row['entities']).strip("[]").replace("'", "").split(",")
        entities = [e.strip() for e in entities if e.strip()]
        
        print(f"\n{'='*60}")
        print(f"Row {idx}: {claim[:70]}...")
        print(f"{'='*60}")
        
        # Test entities
        if entities:
            entity_found = False
            for entity in entities[:2]:  # Test first 2 entities only
                if test_entity_lookup(entity, verbose=True):
                    entity_found = True
            if entity_found:
                success_entities += 1
        
        # Test claim search
        if test_claim_search(claim, verbose=True):
            success_claims += 1
    
    print(f"\n{'='*60}")
    print("📊 Summary:")
    print(f"   Entities found: {success_entities}/10")
    print(f"   Claims with search results: {success_claims}/10")
    print(f"{'='*60}")
