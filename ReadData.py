import readData_AWR1642 as radar
import msvcrt
import ManipuleraData as mainp

# Configurate the serial port
# CLIport, Dataport = radar.serialConfig(radar.configFileName)

# Get the configuration parameters from the configuration file
# configParameters = radar.parseConfigFile(radar.configFileName)

# Main loop
label = 'background'
detObj = {}
frameData = []
currentIndex = 0
j = 0
while True:
    try:
        # Update the data and check if the data is okay
        dataOk, detObj = radar.update(detObj)
        if dataOk:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'1':
                    j += 1
                    print(j)
                    label = 'slideUp'
                elif key == b'2':
                    j += 1
                    print(j)
                    label = 'slideDown'
                elif key == b'3':
                    j += 1
                    print(j)
                    label = 'button'
                elif key == b'4':
                    j += 1
                    print(j)
                    label = 'swipeNext'
                elif key == b'5':
                    j += 1
                    print(j)
                    label = 'swipePrev'
                elif key == b'6':
                    j += 1
                    print(j)
                    label = 'flop'
                elif key == b'c':
                    radar.removeLabel(frameData, label)
                    label = 'background'
                elif key == b'g':
                    label = 'goodBackground'
                else:
                    label = 'background'
                print(f'Current label: {label}')
            detObj['Label'] = label

            # Store the current frame into frameData
            frameData.append(detObj)
            currentIndex += 1

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        filename = input("What's the file name? ([Namn]+[Gest]+[nummer i serie])")
        radar.listOfDictToFile(frameData, filename)
        #create processed path
        path_process="ProcessedData\\{}.csv".format(filename)
        mainp.framedata_to_file(frameData, path_process)
        path_trans="TranslatedData\\{}.csv".format(filename)
        mainp.translateFile(path_process, path_trans)
        break

