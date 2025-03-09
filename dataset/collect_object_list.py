from ai2thor.controller import Controller


controller = Controller(
    agentMode="default",
    visibilityDistance=3.5,
    scene="FloorPlan_Val2_5", #

    # step sizes
    gridSize=0.01,
    snapToGrid=False,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=True,
    renderInstanceSegmentation=True,

    # camera properties
    width=1080,
    height=1080,
    fieldOfView=90
)

objects = controller.last_event.metadata['objects']
root_save_dir = '/home/ztl/deeplearn/vlmaps_ithor/vlmaps_dataset/vlmaps_dataset/FloorPlan_Val2_5/'
for i in range (len(objects)):
    print(objects[i]['objectType'],objects[i]['position'])
    with open(root_save_dir+'log_information.txt', 'a') as file:
        file.write(str(objects[i]['objectType']) + str(objects[i]['position']) + '\n')
# positions = controller.step(
#     action="GetReachablePositions"
# ).metadata["actionReturn"]
# print(positions)