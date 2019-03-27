import readData_AWR1642 as radar
import msvcrt

# Configurate the serial port
#CLIport, Dataport = radar.serialConfig(radar.configFileName)

# Get the configuration parameters from the configuration file
#configParameters = radar.parseConfigFile(radar.configFileName)

# START QtAPPfor the plot
#app = radar.QtGui.QApplication([])

# Set the plot
# pg.setConfigOption('background','w')
# win = pg.GraphicsWindow(title="2D scatter plot")
# p = win.addPlot()
# p.setXRange(-0.5,0.5)
# p.setYRange(0,1.5)
# p.setLabel('left',text = 'Y position (m)')
# p.setLabel('bottom', text= 'X position (m)')
# s = p.plot([],[],pen=None,symbol='o')

# vin = pg.GraphicsWindow(title="Velocity scatter plot")
# q = vin.addPlot()
# q.setXRange(0,2.0)
# q.setYRange(-round(configParameters["maxVelocity"]+0.5),round(configParameters["maxVelocity"]+0.5))
# q.setLabel('left',text = 'V radial vilocity (m/s)')
# q.setLabel('bottom', text= 'R distance (m)')
# t = q.plot([],[],pen=None,symbol='o')

# Main loop
label = 'background'
detObj = {}
frameData = []
currentIndex = 0
j = 0
while True:
    try:

        # elapsed=time.time()-tic
        # print("Tid mellan hämtning", elapsed)
        # tic=time.time()
        # Update the data and check if the data is okay
        dataOk, detObj = radar.update(detObj)
        # print("after update")
        if dataOk:
            # elapsed=time.time()-toc
            # print("Tid för update", elapsed)
            # toc=time.time()
            # print("ok main")
            # Store the current frame into frameData
            # frameData[currentIndex] = detObj
            if msvcrt.kbhit():

                key = msvcrt.getch()
                if key == b'1':
                    label = 'slideUp'
                # elif key == b'2':
                #    label = 'slideDown'
                elif key == b'3':
                    label = 'button'
                elif key == b'4':
                    label = 'swipeNext'
                # elif key == b'5':
                #    label = 'swipePrev'
                # elif key == b'6':
                #    label = 'flop'
                elif key == b'c':
                    radar.removeLabel(frameData, label)
                    label = 'background'
                else:
                    j += 1
                    print(j)
                    label = 'background'
                print(f'Current label: {label}')
            detObj['Label'] = label
            frameData.append(detObj)
            currentIndex += 1

        #
        # Sampling frequency of 30 Hz
        # elapsed=time.time()-tic
        # print("t: ",elapsed)

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        radar.CLIport.write(('sensorStop\n').encode())
        radar.CLIport.close()
        radar.Dataport.close()
        # print(frameData)
        filename = input("What's the file name?")
        # frameDataToFile(frameData,filename)
        radar.listOfDictToFile(frameData, filename)
        # win.close()
        break

