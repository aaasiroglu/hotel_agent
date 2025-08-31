"""
Premium Shoes AI - Simple Database Indexer
==========================================
Simple shoe database indexing using utils functions.
"""

import utils
import services

def main():
    """Main indexing function."""
    print("🚀 PREMIUM SHOES AI - Simple Indexer")
    print("=" * 40)
    
    # Shoe image URLs
    shoe_image_links = [
        # Beymen Collection
        "https://cdn.beymen.com/mnresize/888/-/productimages/jbd3iujk.34u_IMG_02_2110101238375.jpg",
        "https://cdn.beymen.com/mnresize/888/-/productimages/t3g5qmoq.j4r_IMG_08_2110099463476.jpg", 
        "https://cdn.beymen.com/mnresize/888/-/productimages/qmzwjnet.exl_IMG_05_2110099148991.jpg",
        "https://cdn.beymen.com/mnresize/888/-/productimages/reeut1tt.2zp_IMG_02_2110097668217.jpg",
        "https://cdn.beymen.com/mnresize/888/-/productimages/41rujiub.md1_IMG_02_2110101541116.jpg",
        "https://cdn.beymen.com/mnresize/888/-/productimages/baaaqcw1.r45_MP_f3fc6e0c-8aab-4a64-8459-32ab935d1059_2_36339138293301156103500516171_910.jpg",
        "https://cdn.beymen.com/mnresize/888/-/productimages/5la4sefp.se0_IMG_02_8683791756367.jpg",
        
        # Boyner Collection
        "https://statics-mp.boyner.com.tr/mnresize/900/1254/Boynerimages/JAB99022097K-M14-R1_2009336736_0.jpg",
        "https://statics-mp.boyner.com.tr/mnresize/900/1254/Boynerimages/5003248483_271_03.jpg",
        "https://statics-mp.boyner.com.tr/mnresize/900/1254/Boynerimages/5003118355_100_02.jpg",
        "https://statics-mp.boyner.com.tr/mnresize/1100/-/Boynerimages/5000079273_100_01.jpg",
        "https://statics-mp.boyner.com.tr/mnresize/900/1254/Boynerimages/5002989410_100_03.jpg",
        "https://statics-mp.boyner.com.tr/mnresize/1100/-/Boynerimages/3569898124405_2006246297_0.jpg?v=1744700616",
        "https://statics-mp.boyner.com.tr/mnresize/1100/-/Boynerimages/5003242749_001_01.jpg?v=1743082358", 
        "https://statics-mp.boyner.com.tr/mnresize/1100/-/Boynerimages/5003123504_001_01.jpg?v=18017041",    
        "https://statics-mp.boyner.com.tr/mnresize/1100/-/Boynerimages/9301371078638_2006017425_0.jpg?v=1746425281",
        # Lacoste
        "https://akn-lacoste.a-cdn.akinoncloud.com/products/2025/01/24/303084/a369c1b0-45a9-4470-9ea8-ccc1fabe7162_size3840x3840_cropCenter.jpg",
    ]
    
    print(f"📦 Processing {len(shoe_image_links)} shoe images...")
    
    # Process images using utils function
    utils.process_images_to_json_and_insert(shoe_image_links)
    
    # Quick test
    print("\n🧪 Testing search...")
    try:
        results = utils.vector_search_shoes("spor ayakkabı", 2)
        if results:
            print("✅ Search works!")
        else:
            print("❌ Search test failed")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("\n🎯 Ready! Start app: chainlit run main2.py")

if __name__ == "__main__":
    main()
