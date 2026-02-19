import requests
import json

API_URL = "http://localhost:8000/search"

TEST_SET = [
    ("What is Max House, Okhla?", 15),
    ("When was Max Estates established?", 4),
    ("In which region does Max Estates have presence across multiple projects?", 10),
    ("Which project is located on the Noida-Greater Noida Expressway?", 10),
    ("What percentage of Estate 128 is open space (approx)?", 27),
    ("What is the UP RERA registration number for Estate 128?", 27),
    ("How large is 'The Hub' at Estate 128 in sq. ft. (approx)?", 16),
    ("What LEED certification does Max Towers have?", 11),
    ("What event facility does Max Towers provide (seating capacity mentioned)?", 11),
    ("Where is Max Square located?", 19),
    ("How many villas are at 222 Rajpur?", 23),
    ("What is the area of 222 Rajpur (in acres)?", 23),
    ("Who is the artist of 'The Jogger' artwork at Max Square?", 36),
    ("What is the focus of Max India Foundation (one area)?", 43),
    ("What three pillars are emphasized under 'Sustainability and E.S.G.'?", 42),
    ("Which building is IGBC Platinum rated and mentioned for health/well-being?", 11),
    ("Where is Max House Phase 1 located?", 10),
    ("What is the development potential of Max Square Two (approx)?", 34),
    ("In which city is Max Estates' corporate office located (city name)?", 48),
    ("Which foundation has impacted over 19 million people according to the brochure?", 43),
    ("What certifications does Max House have (LEED / IGBC level)?", 15),
    ("Which Max Towers facility seats 374 people?", 11),
    ("Which Max Estates project is described as having a ~11,000 sq. ft. forest?", 19),
    ("How much new development does Max Estates commit to add each year (sq. ft.)?", 34),
    ("Which page/section covers 'Sustainability and E.S.G.'?", 42),
    ("Who is the artist credited for 'Holderstebolde' (the elevator lobby piece)?", 37),
    ("Which upcoming market/area did Max Estates enter with a ~7.15-acre parcel?", 34),
    ("Which project is described in detail on the 'Max Towers' spread?", 11),
    ("What certification is Max Square listed with (IGBC level)?", 46),
    ("Which page lists awards such as 'Emerging Developer of the year - ET Real Estate Awards'?", 45),
    ("Which section explains the role of art at Max Estates?", 35),
    ("Where is Max Estates' corporate office listed in disclaimers?", 48),

]

def evaluate():
    print(f"Running evaluation on {len(TEST_SET)} queries...\n")
    
    top_1_correct = 0
    top_3_correct = 0
    total_latency = 0

    for query, expected_page in TEST_SET:
        try:
            resp = requests.post(API_URL, json={"query": query, "k": 3})
            data = resp.json()
            
            results = data.get("results", [])
            latency = data.get("latency_ms", 0)
            total_latency += latency
            
            print("-" * 50)
            print(f"QUERY: '{query}'")
            print(f"EXPECTED PAGE: {expected_page}")
            print(f"LATENCY: {latency:.2f} ms")
            print("\nRETRIEVED RESULTS:")
            
            found_pages = []
            for i, res in enumerate(results):
                rank = i + 1
                page = res.get("page", "N/A")
                score = res.get("score", 0.0)
                text_preview = res.get("text", "")[:120].replace('\n', ' ')
                
                found_pages.append(page)

                match_marker = " [CORRECT]" if page == expected_page else ""
                print(f"  {rank}. [Page {page}] (Score: {score:.4f}){match_marker}")
                print(f"     Text: {text_preview}...")

            if results and results[0].get("page") == expected_page:
                top_1_correct += 1

            if expected_page in found_pages:
                top_3_correct += 1

        except Exception as e:
            print(f"Error testing query '{query}': {e}")

    # Final Metrics
    if len(TEST_SET) > 0:
        avg_latency = total_latency / len(TEST_SET)
        print("\n")
        print("FINAL SUCCESS METRICS")
        print(f"Average Query Latency: {avg_latency:.2f} ms")
        print(f"Top-1 Accuracy:        {(top_1_correct/len(TEST_SET))*100:.1f}%")
        print(f"Top-3 Accuracy:        {(top_3_correct/len(TEST_SET))*100:.1f}%")

if __name__ == "__main__":
    evaluate()