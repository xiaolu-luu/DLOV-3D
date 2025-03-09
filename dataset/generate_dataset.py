import os
import numpy as np

def generate_object_goal_navigation(language_instr):
    import openai

    openai_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "object:[The mug which next to The computer, The desk which has a red apple on it, The table with The notebooks, Phone on The table,\
                            The table with the bowl, The notebook on The table, The fork on the desk, Green sofa, The desk which has some book on it, The sofa with the red pillow,\
                            The vase which next to the plant, The purple pillow on The sofa, The sofa with The laptop, The books on the sofa, The chair in front of the table, The laptop on the chair,\
                            desk, box, red chair, coffee machine, The green apple which next to the computer,The chair which has a basketball under it, The chair with the newspaper, The white chair in front of The laptop,\
                            The bed with yellow quilt, cell phone on the bed, The desk with a computer, The book next to the computer,The house plant on the desk,\
                            The mug on the desk, The laptop on the desk, The black wooden chair, The white chair with a scarf, The table, window, The shelf with potted plant,\
                            The yellow pillow, The desk with books on it, The hard wooden chair, basketball under the chair, garbage can, baseball bat]\
                            task:Generate 10 robot navigation instructions according to The list of items provided above, each instruction contains 4 sub-commands,The logical relationship between The four sub-commands conforms to The human operation,At The same time, use all The items in The list as much as possible."
            },
            {
                "role": "assistant",
                "content": "1.Go to the mug which next to The computer, Then,go to the chair which has a basketball under it, then go to the red chair, finally ,navigate to the house plant.\n \
                            2.Go to the desk which has a red apple on it, Then approach the sofa with the laptop, later navigate to the table and the bed with yellow quilt.\n \
                            3.Go to the desk with books on it, Then,go to the house plant, recognize the laptop on the sofa and Scan the surrounding area for the green apple next to the computer.\n \
                            4.Go to the desk with a computer, Then,locate the hard wooden chair nearby, move to the coffee machine and find a clean mug on the desk.\n \
                            5.Go to the house plant immediately, later find the book on the sofa, then go to the white chair with a scarf and finally go to a nearby window.\n \
                            6.Move to the table with The notebooks, recognize the phone on the table, come to the table with the bowl and go to the green sofa.\n \
                            7.Move to the desk which has some book on it, go to the sofa with the red pillow, find the vase which next to the plant and go to the chair with the newspaper.\n \
                            8.Turn around and find the yellow pillow, go to the shelf with potted plant, come to the garbage can and go to the baseball bat.\n \
                            9.Move to the bed with yellow quilt, go to the green sofa, come to the chair which has a basketball under it and go to the desk with books on it.\n \
                            10.Approach to the mug on the desk, find the purple pillow on the sofa, come to the chair in front of the table and navigate to the shelf with potted plant.\n "
            },
            {
                "role": "user",
                "content": language_instr+"\n Generate 10 robot navigation instructions according to The list of items provided above, each instruction contains 4 sub-commands,The logical relationship between The four sub-commands conforms to The human operation,At The same time, use all The items in The list as much as possible."
            }
        ],
        max_tokens=4096,
    )

    text = response.choices[0].message.content
    return [x.strip() for x in text.split("\n")]
def main(language_instr,root_save_dir):
    language_instr_str = str(language_instr)
    navigation_instructions = generate_object_goal_navigation(language_instr_str)
    with open(root_save_dir+'log_information.txt', 'a') as file:
        file.write(language_instr_str + '\n')
        for item in navigation_instructions:
        # 将每个元素写入文件，并在每个元素后添加换行符
            file.write(item + '\n')
    return 0

if __name__ == "__main__":
    root_save_dir = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan_Val2_4/'
    language_instr = [
        'green apple',
        'red arm chair',
        'red sofa',
        'white arm chair',
        'baseball bat',
        'basketball',
        'books',
        'box',
        'The bed which has a laptop on it',
        'The desk with the house plant',
        'green garbage can',
        'house plant',
        'laptop',
        'mug',
        'sofa with basketball',
        'bed with blue pillow',
        'blue pillow',
        'teddy bear',
        'television',
        'gold statue',
                 ]
    # language_instr = str(item_list)
    print(len(language_instr))
    main(language_instr,root_save_dir)
