from PIL import Image
from io import BytesIO
from vertexai.preview.generative_models import Image as VertexImage
from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex



prompt = {
    "intro": """You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page screenshot: This is a screenshot of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
The observation, which lists the IDs of all interactable elements on the current web page with their text content if any, in the format [id] [tagType] [text content]. tagType is the type of the element, such as button, link, or textbox. text content is the text content of the element. For example, [1234] [button] ['Add to Cart'] means that there is a button with id 1234 and text content 'Add to Cart' on the current web page. [] [StaticText] [text] means that the element is of some text that is not interactable.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.

Homepage:
If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.
http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.""",
    "examples": [
        (
            """OBSERVATION:
[31] [IMG] [Image, description: hp fx-7010dn fax machine, url: http://ec2-3-13-232-171.us-east-2.compute.amazonaws.com:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B08GKZ3ZKD.0.jpg]
[32] [A] [HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)]
[] [StaticText] [$279.49]
[33] [BUTTON] [Add to Cart]
[34] [A] [Add to Wish List]
[35] [A] [Add to Compare]
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine?
PREVIOUS ACTION: None""",
            "Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```",
            "som_examples/som_example1.png"
        ),
        (
            """OBSERVATION:
[] [StaticText] [/f/food]
[] [StaticText] [[homemade] Obligatory Halloween Pumpkin Loaf!  Submitted by    kneechalice t3_yid9lu   1 year ago]
[9] [IMG] []
[] [StaticText] [Submitted by   kneechalice t3_yid9lu   1 year ago]
[10] [A] [kneechalice]
[11] [A] [45 comments]
[] [StaticText] [[I ate] Maple Pecan Croissant  Submitted by    AccordingtoJP   t3_y3hrpn   1 year ago]
[14] [IMG] []
[] [StaticText] [Submitted by   AccordingtoJP   t3_y3hrpn   1 year ago]
[15] [A] [AccordingtoJP]
[16] [A] [204 comments]
URL: http://reddit.com
OBJECTIVE: Tell me what the top comment on the croissant post says.
PREVIOUS ACTION: None""",
            "Let's think step-by-step. This page has a post titled '[I ate] Maple Pecan Croissant', which is the post mentioned in the objective. In order to find the top comment, I will navigate into the comments section of the post. In summary, the next action I will perform is ```click [11]```",
            "som_examples/som_example2.png"
        ),
        (
            """OBSERVATION:
[] [StaticText] [What are you looking for today?]
[5] [INPUT] []
[6] [SELECT] [Select a category]
[7] [BUTTON] [Search]
[] [StaticText] [Latest Listings]
[] [StaticText] [Atlas Powered Audio System w/ Tripod   150.00 $    Music instruments   Borough of Red Lion  (Pennsylvania) 2023/11/16]
[8] [IMG] [Atlas Powered Audio System w/ Tripod]
[9] [A] [Atlas Powered Audio System w/ Tripod]
[] [StaticText] [150.00 $]
[] [StaticText] [Neptune Gaming Console 350.00 $    Video gaming    Pennwyn  (Pennsylvania) 2023/11/16]
[10] [IMG] [Neptune Gaming Console]
[11] [A] [Neptune Gaming Console]
[] [StaticText] [350.00 $]
URL: http://classifieds.com
OBJECTIVE: Help me find the cheapest dark colored guitar.
PREVIOUS ACTION: None""",
            "Let's think step-by-step. The objective is to find the cheapest dark colored guitar on the site. The site has a search box whose ID is [5]. I can search for guitars by entering \"guitar\". I can submit this by pressing the Enter afterwards. In summary, the next action I will perform is ```type [5] [guitar] [1]```",
            "som_examples/som_example3.png"
        ),
    ],
    "template": """OBSERVATION: {observation}
URL: {url}
OBJECTIVE: {objective}
PREVIOUS ACTION: {previous_action}""",
    "meta_data": {
        "observation": "image_som",
        "action_type": "som",
        "keywords": ["url", "objective", "observation", "previous_action"],
        "prompt_constructor": "MultimodalCoTPromptConstructor",
        "answer_phrase": "In summary, the next action I will perform is",
        "action_splitter": "```"
    },
}




if __name__ == "__main__":
    model = GenerativeModel("gemini-pro-vision")
    intro = prompt["intro"]
    examples = prompt["examples"]
    obs = """
[] [StaticText] [My Account]
[1] [A] [My Account]
[] [StaticText] [My Wish List]
[2] [A] [My Wish List]
[] [StaticText] [Sign Out]
[3] [A] [Sign Out]
[] [StaticText] [Welcome, Emma Lopez!]
[5] [IMG] [one_stop_market_logo, description: one stop market logo, url: http://localhost:7770/media/logo/websites/1/image_15__1.png]
[6] [A] [My Cart                                                    6                                                                    6                items]
[] [StaticText] [6]
[] [StaticText] [6                items]
[] [StaticText] [items]
[] [StaticText] [Search]
[7] [INPUT] []
[8] [A] [Advanced Search]
[9] [BUTTON] [Search]
[10] [UL] [Beauty & Personal CareOral CareToothbrushes & AccessoriesDental Floss & PicksOrthodontic SuppliesChildren's Dental CareOral Pain ReliefToothpasteTeeth WhiteningBreath FreshenersDenture CareTongue Clea]
[11] [A] [Beauty & Personal Care]
[12] [A] [Sports & Outdoors]
[13] [A] [Clothing, Shoes & Jewelry]
[14] [A] [Home & Kitchen]
[15] [A] [Office Products]
[16] [A] [Tools & Home Improvement]
[17] [A] [Health & Household]
[18] [A] [Patio, Lawn & Garden]
[19] [A] [Electronics]
[20] [A] [Cell Phones & Accessories]
[21] [A] [Video Games]
[22] [A] [Grocery & Gourmet Food]
[] [StaticText] [One Stop Market]
[23] [IMG] [Image, description: a package of gingerbread house cookies, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B08PCSHBXY.0.jpg]
[24] [A] [Pre-baked Gingerbread House Kit Value Pack, 17 oz., Pack of 2, Total 34 oz.]
[] [StaticText] [Rating:]
[] [StaticText] [20%]
[25] [A] [1                 Review]
[] [StaticText] [$19.99]
[26] [BUTTON] [Add to Cart]
[27] [A] [Add to Wish List]
[28] [A] [Add to Compare]
[29] [IMG] [Image, description: energy drink with berries and grapes, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B00CPTR7WS.0.jpg]
[30] [A] [V8 +Energy, Healthy Energy Drink, Steady Energy from Black and Green Tea, Pomegranate Blueberry, 8 Ounce Can ,Pack of 24]
[] [StaticText] [57%]
[31] [A] [12                 Reviews]
[] [StaticText] [$14.47]
[32] [BUTTON] [Add to Cart]
[33] [A] [Add to Wish List]
[34] [A] [Add to Compare]
[35] [IMG] [Image, description: orange vanilla coffee flavoring, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B0040WHKIY.0.jpg]
[36] [A] [Elmwood Inn Fine Teas, Orange Vanilla Caffeine-free Fruit Infusion, 16-Ounce Pouch]
[] [StaticText] [95%]
[37] [A] [4                 Reviews]
[] [StaticText] [$19.36]
[38] [BUTTON] [Add to Cart]
[39] [A] [Add to Wish List]
[40] [A] [Add to Compare]
[41] [IMG] [Image, description: a red and yellow cake with roses and yellow candy, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B08743KW8M.0.jpg]
[42] [A] [Belle Of The Ball Princess Sprinkle Mix| Wedding Colorful Sprinkles| Cake Cupcake Cookie Sprinkles| Ice cream Candy Sprinkles| Yellow Gold Red Royal Red Rose Icing Flowers Decorating Sprinkles, 8OZ]
[] [StaticText] [63%]
[43] [A] [12                 Reviews]
[] [StaticText] [$23.50]
[44] [BUTTON] [Add to Cart]
[45] [A] [Add to Wish List]
[46] [A] [Add to Compare]
[47] [IMG] [Image, description: so delicious coconut whip, coconut whip, coconut whip, coconut whip, coconut whip, coconut whip, coconut whip, coconut whip, coconut whip, coconut whip,, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B01N1QRJNG.0.jpg]
[48] [A] [So Delicious Dairy Free CocoWhip Light, Vegan, Non-GMO Project Verified, 9 oz. Tub]
[] [StaticText] [78%]
[49] [A] [12                 Reviews]
[] [StaticText] [$15.62]
[50] [BUTTON] [Add to Cart]
[51] [A] [Add to Wish List]
[52] [A] [Add to Compare]
[53] [IMG] [Image, description: korean fried rice - 2 packs, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B07Y4MC6M5.0.jpg]
[54] [A] [Cheongeun Sweet Potato Starch Powder 500g, 2ea(Sweet Potato 55%, Corn 45%)]
[] [StaticText] [$34.00]
[55] [BUTTON] [Add to Cart]
[56] [A] [Add to Wish List]
[57] [A] [Add to Compare]
[58] [IMG] [Image, description: a bottle of ginger ale on a white background, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B071KC37VD.0.jpg]
[59] [A] [Q Mixers Premium Ginger Ale: Real Ingredients & Less Sweet, 6.7 Fl Oz (24 Bottles)]
[] [StaticText] [88%]
[60] [A] [12                 Reviews]
[] [StaticText] [$68.50]
[61] [BUTTON] [Add to Cart]
[62] [A] [Add to Wish List]
[63] [A] [Add to Compare]
[64] [IMG] [Image, description: kraft stove top turkey & cranberry, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B004U5F23G.0.jpg]
[65] [A] [Stove Top Turkey Stuffing Mix (12 oz Boxes, Pack of 2)]
[] [StaticText] [85%]
[66] [A] [12                 Reviews]
[] [StaticText] [$8.49]
[67] [BUTTON] [Add to Cart]
[68] [A] [Add to Wish List]
[69] [A] [Add to Compare]
[70] [IMG] [Image, description: a beige powder on a white background, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B074DBMG66.0.jpg]
[71] [A] [Briess DME - Pilsen Light - 1 lb Bag]
[] [StaticText] [$12.99]
[72] [BUTTON] [Add to Cart]
[73] [A] [Add to Wish List]
[74] [A] [Add to Compare]
[75] [IMG] [Image, description: tony chachere's more spice seasoning, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B001Q1J3QY.0.jpg]
[76] [A] [Tony Chachere's More Spice Creole Seasoning - 14 oz]
[] [StaticText] [75%]
[77] [A] [12                 Reviews]
[] [StaticText] [$7.92]
[78] [BUTTON] [Add to Cart]
[79] [A] [Add to Wish List]
[80] [A] [Add to Compare]
[81] [IMG] [Image, description: a plate of lobster pies with carrots and peas, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B0793RMRGT.0.jpg]
[82] [A] [Lobster Cobbler Pot Pie - Gourmet Frozen Seafood Appetizers (Set of 8 Trays)]
[] [StaticText] [$411.76]
[83] [BUTTON] [Add to Cart]
[84] [A] [Add to Wish List]
[85] [A] [Add to Compare]
[86] [IMG] [Image, description: crunchy rollers - pack of 6, url: http://localhost:7770/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B00J7HXXGA.0.jpg]
[87] [A] [Crunchy Rice Rollers - Gluten Free - Vegan - 3.5 oz Individual Packs (4 Packs of 8 Rollers)]
[] [StaticText] [83%]
[88] [A] [12                 Reviews]
[] [StaticText] [$11.50]
[89] [BUTTON] [Add to Cart]
[90] [A] [Add to Wish List]
[91] [A] [Add to Compare]
[] [StaticText] [Items 1 to 12 of 24 total]
    """
    page_screenshot_img = Image.open('current.png')

    message = [
        intro,
        "Here are a few examples:",
    ]
    for (x, y, z) in examples:
        example_img = Image.open(z)
        message.append(f"Observation\n:{x}\n")
        message.extend(
            [
                "IMAGES:",
                "(1) current page screenshot:",
                pil_to_vertex(example_img),
            ]
        )
        message.append(f"Action: {y}")
    message.append("Now make prediction given the observation")
    current = prompt["template"].format(
        objective="What is the price range for products in the first row of this page?",
        url="http://localhost:7770/",
        observation=obs,
        previous_action="None",
    )
    message.append(f"Observation\n:{current}\n")
    message.extend(
        [
            "IMAGES:",
            "(1) current page screenshot:",
            pil_to_vertex(page_screenshot_img),
        ]
    )
    message.append("Action:")
    print([type(x) for x in message])

    safety_config = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    response = model.generate_content(
        message,
        generation_config=dict(
            candidate_count=1,
            max_output_tokens=384,
            top_p=1.0,
            temperature=0.9,
        ),
        # safety_settings=safety_config,
    )
    answer = response.text
    print(answer)

