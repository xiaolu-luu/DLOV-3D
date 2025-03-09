import os
import openai


def parse_object_goal_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    openai_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_key
    question = f"""
    I: go to the kitchen and then go to the toilet. A: kitchen, toilet
    I: go to the chair and then go to another chair. A: chair, chair
    I: navigate to the green sofa and turn right and find several chairs, finally go to the painting. A: green sofa, chairs, painting
    I: approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen. A: window, television, oven, kitchen
    I: walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed. A: plant, table, bedroom, bed
    I: go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room. A: stairs, room, book shelf, table, next room
    I: Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances. A: table, cushion, column, appliances.
    I: Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table. A: chair, table
    I: {language_instr}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"].strip()
    print("landmarks: ", result)
    return [x.strip() for x in result.split(",")]

def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    openai_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "go to the kitchen and then go to the toilet",
            },
            {
                "role": "assistant",
                "content": "kitchen, toilet"
            },
            {
                "role": "user",
                "content": "go to the chair and then go to another chair"
            },
            {
                "role": "assistant",
                "content": "chair, chair"
            },
            {
                "role": "user",
                "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"
            },
            {
                "role": "assistant",
                "content": "green sofa, chairs, painting"
            },
            {
                "role": "user",
                "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"
            },
            {
                "role": "assistant",
                "content": "window, television, oven, kitchen"
            },
            {
                "role": "user",
                "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"
            },
            {
                "role": "assistant",
                "content": "plant, table, bedroom, bed"
            },
            {
                "role": "user",
                "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"
            },
            {
                "role": "assistant",
                "content": "stairs, room, book shelf, table, next room"
            },
            {
                "role": "user",
                "content": "move to the bed with yellow quilt, then turn around and find cell phone on the bed, later stand next to the chair which has a basketball under it, finally navigate to the desk with a computer"
            },
            {
                "role": "assistant",
                "content": "the bed with yellow quilt, cell phone on the bed, the chair which has a basketball under it, the desk with a computer"
            },
            {
                "role": "user",
                "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"
            },
            {
                "role": "assistant",
                "content": "chair, table"
            },
            {
                "role": "user",
                "content": "Go to the black wooden chair, Then,go to the white chair with a scarf, the table and the black wooden chair.",
            },
            {
                "role": "assistant",
                "content": "the black wooden chair, the white chair with a scarf, table, the black wooden chair"
            },
            {
                "role": "user",
                "content": "walk to take the apple next to the computer first, turn around and come back to the chair, go to the bed with yellow quilt, and stand next to the red chair"
            },
            {
                "role": "assistant",
                "content": "the apple next to the computer, the chair, the bed with yellow quilt, the red chair"
            },
            {
                "role": "user",
                "content": "Move to the chair, then navigate to the garbage can, move to the house plant, then find newspaper on the chair"
            },
            {
                "role": "assistant",
                "content": "the chair, the garbage can, the house plant, newspaper on the chair"
            },
            {
                "role": "user",
                "content": "approach the window in front, turn around and go to the shelf with potted plant, and finally go to the desk with books on it"
            },
            {
                "role": "assistant",
                "content": "window, the shelf with potted plant, the desk with books on it"
            },
            {
                "role": "user",
                "content": "navigate to the green sofa and find Remote Control next to the television, finally go to the white chair in front of the laptop"
            },
            {
                "role": "assistant",
                "content": "green sofa, Remote Control next to the television, the white chair in front of the laptop"
            },
            {
                "role": "user",
                "content": "go to the desk lamp and then take the newspaper on the chair"
            },
            {
                "role": "assistant",
                "content": "desk lamp, the newspaper on the chair"
            },
            {
                "role": "user",
                "content": "Move to the yellow pillow, then navigate to the desk with books on it, move to the hard wooden chair, then find basketball under the chair"
            },
            {
                "role": "assistant",
                "content": "the yellow pillow, the desk with books on it, the hard wooden chair, basketball under the chair"
            },
            {
                "role": "user",
                "content": language_instr
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    return [x.strip() for x in text.split(",")]

def parse_object_goal_instruction_s(language_instr):
    import openai

    openai_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are now a home service robot, please output the navigation target according to the navigation instructions entered by the user",
            },
            {
                "role": "assistant",
                "content": "Ok, I see what you mean. I will output the result as required"
            },
            {
                "role": "user",
                "content": "The mug which next to the computer,the desk which has a red apple on it, the table with the notebooks, phone on the table",
            },
            {
                "role": "assistant",
                "content": "mug, desk, table, phone"
            },
            {
                "role": "user",
                "content": "the table with the bowl, the notebook on the table, the fork on the desk, green sofa"
            },
            {
                "role": "assistant",
                "content": "table, notebook, fork, green sofa"
            },
            {
                "role": "user",
                "content": "the desk which has some book on it, the sofa with the red pillow, the vase which next to the plant, the purple pillow on the sofa"
            },
            {
                "role": "assistant",
                "content": "desk, sofa, vase, purple pillow"
            },
            {
                "role": "user",
                "content": "the sofa with the laptop, the books on the sofa, the chair in front of the table, the laptop on the chair"
            },
            {
                "role": "assistant",
                "content": "sofa, books, chair, laptop"
            },
            {
                "role": "user",
                "content": "desk, box, red chair, coffee machine"
            },
            {
                "role": "assistant",
                "content": "desk, box, red chair, coffee machine"
            },
            {
                "role": "user",
                "content": "The green apple which next to the computer,the chair which has a basketball under it, the chair with the newspaper, the white chair in front of the laptop"
            },
            {
                "role": "assistant",
                "content": "green apple, chair, chair, white chair"
            },
            {
                "role": "user",
                "content": "the bed with yellow quilt, cell phone on the bed, the chair which has a basketball under it, the desk with a computer"
            },
            {
                "role": "assistant",
                "content": "bed, cell phone, chair, desk"
            },
            {
                "role": "user",
                "content": "the book next to the computer,the house plant on the desk, the mug on the desk, the laptop on the desk"
            },
            {
                "role": "assistant",
                "content": "book, house plant, mug, laptop"
            },
            {
                "role": "user",
                "content": "the black wooden chair, the white chair with a scarf, the table ,the black wooden chair",
            },
            {
                "role": "assistant",
                "content": "black wooden chair, white chair, table, black wooden chair"
            },
            {
                "role": "user",
                "content": "the apple next to the computer,chair, the bed with yellow quilt, the red chair"
            },
            {
                "role": "assistant",
                "content": "apple, chair, bed, red chair"
            },
            {
                "role": "user",
                "content": "the chair, the garbage can, the house plant, newspaper on the chair"
            },
            {
                "role": "assistant",
                "content": "chair, garbage can, house plant, newspaper"
            },
            {
                "role": "user",
                "content": "window, the shelf with potted plant, the desk with books on it"
            },
            {
                "role": "assistant",
                "content": "window, shelf, desk"
            },
            {
                "role": "user",
                "content": "the green sofa ,Remote Control next to the television,the white chair in front of the laptop"
            },
            {
                "role": "assistant",
                "content": "green sofa, Remote Control, white chair"
            },
            {
                "role": "user",
                "content": "the desk lamp ,the newspaper on the chair"
            },
            {
                "role": "assistant",
                "content": "desk lamp, newspaper"
            },
            {
                "role": "user",
                "content": "the yellow pillow,the desk with books on it,the hard wooden chair,basketball under the chair"
            },
            {
                "role": "assistant",
                "content": "yellow pillow, desk, hard wooden chair, basketball"
            },
            {
                "role": "user",
                "content": language_instr
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    return [x.strip() for x in text.split(",")]


def parse_spatial_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        question = f"""
# move a bit to the right of the refrigerator.
robot.move_to_right('refrigerator')
###
# move in between the couch and bookshelf.
robot.move_in_between('couch', 'bookshelf')
###
# move in the middle of the stairs and oven.
robot.move_in_between('stairs', 'oven')
###
# face the toilet.
robot.face('toilet')
###
# move to the south side of the table
robot.move_south('table')
###
# move to the west of the chair
robot.move_west('chair')
###
# move to the east of the chair
robot.move_east('chair')
###
# turn left 60 degrees
robot.turn(-60)
###
# turn right 20 degrees
robot.turn(20)
###
# find any chairs in the environment
robot.move_to_object('chair')
###
# come to the table
robot.move_to_object('table')
###
# with the television on your left
robot.with_object_on_left('television')
###
# with the toilet to your right, turn left 30 degrees
robot.with_object_on_right('toilet')
robot.turn(-30)
###
# with the toilet to your left, turn right 30 degrees
robot.with_object_on_left('toilet')
robot.turn(30)
###
# with the television behind you
robot.face('television')
robot.turn(180)
###
# go to the south of the table in front of you, with the table on your left, turn right 45 degrees
robot.move_south('table')
robot.with_object_on_left('table')
robot.turn(45)
###
# move forward for 3 meters
robot.move_forward(3)
###
# move right 2 meters
robot.turn(90)
robot.move_forward(2)
###
# move left 3 meters
robot.turn(-90)
robot.move_forward(3)
###
# move back and forth between the chair and the table 3 times
pos1 = robot.get_pos('chair')
pos2 = robot.get_pos('table')
for i in range(3):
    robot.move_to(pos1)
    robot.move_to(pos2)
###
# go to the television
robot.move_to_object('television')
###
# monitor the chair from 4 view points
contour = robot.get_contour('chair')
np.random.shuffle(contour)
for pos in contour[:4]:
    robot.move_to(pos)
    robot.face('chair')
###
# navigate to 3 meters right of the table
robot.move_to_right('table')
robot.face('table')
robot.turn(180)
robot.move_forward(3)
###
# turn west
robot.turn_absolute(-90)
###
# turn east
robot.turn_absolute(90)
###
# turn south
robot.turn_absolute(180)
###
# turn north
robot.turn_absolute(0)
###
# turn east and then turn left 90 degrees
robot.turn_absolute(90)
robot.turn(-90)
###
# move 3 meters north of the chair
robot.move_north('chair')
robot.turn_absolute(0)
robot.move_forward(3)
###
# move 1 meters south of the chair
robot.move_south('chair')
robot.turn_absolute(180)
robot.move_forward(3)
###
# move 3 meters west
robot.turn_absolute(-90)
robot.move_forward(3)
# {lang}
    """
        print("lang: ", lang)
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=question,
            max_tokens=400,
            temperature=0.0,
            stop="###",
        )
        result = response["choices"][0]["text"].strip()
        if result:
            results += result + "\n"
        print(result)
    print("generated code")
    print(results)
    return results

def parse_spatial_instruction(language_instr):
    import openai

    openai_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": "move a bit to the right of the refrigerator"},
                {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
                {"role": "user", "content": "move in between the couch and bookshelf"},
                {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
                {"role": "user", "content": "move in the middle of the stairs and oven"},
                {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
                {"role": "user", "content": "face the toilet"},
                {"role": "assistant", "content": "robot.face('toilet')"},
                {"role": "user", "content": "move to the south side of the table"},
                {"role": "assistant", "content": "robot.move_south('table')"},
                {"role": "user", "content": "move to the west of the chair"},
                {"role": "assistant", "content": "robot.move_west('chair')"},
                {"role": "user", "content": "move to the east of the chair"},
                {"role": "assistant", "content": "robot.move_east('chair')"},
                {"role": "user", "content": "turn left 60 degrees"},
                {"role": "assistant", "content": "robot.turn(-60)"},
                {"role": "user", "content": "turn right 20 degrees"},
                {"role": "assistant", "content": "robot.turn(20)"},
                {"role": "user", "content": "find any chairs in the environment"},
                {"role": "assistant", "content": "robot.move_to_object('chair')"},
                {"role": "user", "content": "come to the table"},
                {"role": "assistant", "content": "robot.move_to_object('table')"},
                {"role": "user", "content": "with the television on your left"},
                {"role": "assistant", "content": "robot.with_object_on_left('television')"},
                {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
                {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
                {"role": "user", "content": "with the television behind you"},
                {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
                {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
                {"role": "assistant", "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
                {"role": "user", "content": "move forward for 3 meters"},
                {"role": "assistant", "content": "robot.move_forward(3)"},
                {"role": "user", "content": "move right 2 meters"},
                {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
                {"role": "user", "content": "move left 3 meters"},
                {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
                {"role": "assistant", "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
                {"role": "user", "content": "go to the television"},
                {"role": "assistant", "content": "robot.move_to_object('television')"},
                {"role": "user", "content": "monitor the chair from 4 view points"},
                {"role": "assistant", "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
                {"role": "user", "content": "navigate to 3 meters right of the table"},
                {"role": "assistant", "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
                {"role": "user", "content": "turn west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)"},
                {"role": "user", "content": "turn east"},
                {"role": "assistant", "content": "robot.turn_absolute(90)"},
                {"role": "user", "content": "turn south"},
                {"role": "assistant", "content": "robot.turn_absolute(180)"},
                {"role": "user", "content": "turn north"},
                {"role": "assistant", "content": "robot.turn_absolute(0)"},
                {"role": "user", "content": "turn east and then turn left 90 degrees"},
                {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
                {"role": "user", "content": "move 3 meters north of the chair"},
                {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 1 meters south of the chair"},
                {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 3 meters west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": lang},
            ],
            max_tokens=300,
        )
        text = response.choices[0].message.content
        if text:
            results += text + "\n"
    return text
if __name__ == '__main__':
    # text = parse_spatial_instruction("go to the sofa, turn right and move in between the table and the chair, and then move back and forth to the keyboard and the screen twice")
    text = parse_object_goal_instruction('find cell phone on the bed, then pick up the phone and turn around and move to the bed with yellow quilt , later stand next to the red chair which has a laptop on it, finally navigate to the wooden desk with a book')
    print(text)
    text2 = parse_object_goal_instruction_s(str(text))
    print(text2)