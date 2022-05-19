def get_meta_data():
    return {
        "sustenance": {
            "filename": "sustenance_pois.json",
            "query": """
                [out:json]; 
                area[name = "Leipzig"]->.a;
                (   
                    node(area.a)[amenity=bar];
                    node(area.a)[amenity=restaurant]; 
                    node(area.a)[amenity=pub];
                    node(area.a)[amenity=ice_cream];
                    node(area.a)[amenity=food_court];
                    node(area.a)[amenity=fast_food];
                    node(area.a)[amenity=biergarten];
                    node(area.a)[amenity=cafe];
                ); 
                out;  
            """,
        },
        "public_transport": {
            "filename": "public_transport_pois.json",
            "query": """ [out:json]; 
                area[name = "Leipzig"]->.a;
                (   
                    node(area.a)[public_transport=station];
                    node(area.b)[public_transport=station];
                    node(area.c)[public_transport=station];
                ); 
                out; """,
        },
        "education": {
            "filename": "education_pois.json",
            "query": """ [out:json]; 
                area[name = "Leipzig"]->.a;
                (   
                    node(area.a)[amenity=college];
                    node(area.a)[amenity=library];
                    node(area.a)[amenity=driving_school];
                    node(area.a)[amenity=language_school];
                    node(area.a)[amenity=music_school];
                    node(area.a)[amenity=school];
                    node(area.a)[amenity=university];

                ); 
                out; """,
        },
        "arts_and_culture": {
            "filename": "arts_and_culture_pois.json",
            "query": """ [out:json]; 
                area[name = "Leipzig"]->.a;
                (   
                    node(area.a)[amenity=arts_centre];
                    node(area.a)[amenity=theatre];
                    node(area.a)[amenity=cinema];
                ); 
                out;""",
        },
        "sports": {
            "filename": "sports.json",
            "query": """[out:json]; 
                area[name = "Leipzig"]->.a;
                (
                    node(area.a)[leisure=fitness_station];
                    node(area.a)[leisure=fitness_centre];                
                    node(area.a)[leisure=sports_centre];
                    node(area.a)[leisure=swimming_area];
                    node(area.a)[leisure=swimming_pool];
                ); 
                out;""",
        },
    }
