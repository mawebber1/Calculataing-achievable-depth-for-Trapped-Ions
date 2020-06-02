from IPython.display import clear_output
import time
import math
import random
import numpy as np 
import pygame

#define device with size and resolution
def XJunction(numberx, numbery, resolution):
    zeros = [ [0] * numbery*resolution for _ in range(numberx*resolution)]
    for i in range (0,numberx):
        for j in range (0,numbery*resolution):
            zeros[i*resolution+(int(resolution/2))][j]=1
    for i in range (0,numbery):
        for j in range (0,numberx*resolution):
            zeros[j][i*resolution+(int(resolution/2))]=1
    return zeros


def draw(array):
    for row in array:
        print(' '.join([str(elem) for elem in row]))

#defines allowable movement direction (lane system), up,right,down,left
def createMoveArray(array):
    movearray=array
    for j in range(0,len(array)):
        for i in range(0,len(array[0])):
            movearray[j][i]=[0,0,0,0]
            
    #assign everything a direction
    for j in range(0,numberx):
        for i in range(0,len(array[0])):
            movearray[resolution*j+(int(resolution/2))][i]=[0,1,0,1]
    for j in range(0,numbery):
        for i in range(0,len(array)):
            movearray[i][resolution*j+(int(resolution/2))]=[1,0,1,0]

    #assign perimiter directions
    for i in range((int(resolution/2)),len(array[0])-(int(resolution/2))):
        movearray[(int(resolution/2))][i]=[0,1,0,0]
    for i in range((int(resolution/2)),len(array)-(int(resolution/2))):
        movearray[i][len(array[0])-(int(resolution/2)+1)]=[0,0,1,0]
    for i in range((int(resolution/2)),len(array[0])-(int(resolution/2))):
        movearray[len(array)-(int(resolution/2))-1][i]=[0,0,0,1]
    for i in range((int(resolution/2)),len(array)-(int(resolution/2))):
        movearray[i][(int(resolution/2))]=[1,0,0,0]

    #assign interior directoins
    for j in range(1,numberx-1):
        for i in range((int(resolution/2)),len(array[0])-(int(resolution/2))):
            if j%2==0:
                movearray[resolution*j+(int(resolution/2))][i]=[0,1,0,0]
            else:
                movearray[resolution*j+(int(resolution/2))][i]=[0,0,0,1]
    for j in range(1,numbery-1):
        for i in range((int(resolution/2)),len(array)-(int(resolution/2))):
            if j%2==0:
                movearray[i][resolution*j+(int(resolution/2))]=[1,0,0,0]
            else:
                movearray[i][resolution*j+(int(resolution/2))]=[0,0,1,0]

    #assign middle node directions
    for i in range(0,numberx):
        for j in range(0,numbery):
            if movearray[resolution*i+(int(resolution/2)+1)][resolution*j+(int(resolution/2))][2]==1:
                movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2))][2]=1
            if movearray[resolution*i+(int(resolution/2)-1)][resolution*j+(int(resolution/2))][0]==1:
                movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2))][0]=1
            if movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2)+1)][1]==1:
                movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2))][1]=1
            if movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2)-1)][3]==1:
                movearray[resolution*i+(int(resolution/2))][resolution*j+(int(resolution/2))][3]=1
    return movearray


def distance(posx1,posy1,posx2,posy2):
    return ((posx1-posx2)**2+(posy1-posy2)**2)**0.5

def manhattandistance(x1,y1,x2,y2):
    return abs(x1-x2)+abs(y1-y2)


 
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (30,144,255)
ORANGE= (255, 165, 0) 

#determines whether an ion needs to move based on if there are other ions behind it (using the movement direction array)
def need2move(array,ix,iy,movearray,pairdestx,pairdesty,fearparam,propert,destx,desty):
    a=nearestintnode(array,ix,iy) 
    fearparamnew=fearparam
    if (abs(a[0]-ix)==1 and abs(a[1]-iy)==0) or (abs(a[0]-ix==0) and abs(a[1]-iy)==1) or (abs(a[0]-ix)==0 and abs(a[1]-iy)==0): 
        fearparamnew=1
    if (abs(a[0]-ix)==2 and abs(a[1]-iy)==0) or (abs(a[0]-ix==0) and abs(a[1]-iy)==2): 
        #print("did this 1!")
        return 1
    if propert!=2 and ((abs(a[0]-ix)==1 and abs(a[1]-iy)==0) or (abs(a[0]-ix==0) and abs(a[1]-iy)==1) or (abs(a[0]-ix==0) and abs(a[1]-iy)==0)): 
        #print("did this 2!")
        return 1
    if propert==2 and abs(ix-destx)==0 and abs(iy-desty)==0 and (abs(a[0]-destx)>=2):
        return 0
    
    #up
    if movearray[ix][iy][0]==1:
        for i in range(ix+1,min(ix+int(fearparamnew)+1,len(array)-int(resolution/2))):
            if (array[i][iy]==3 or array[i][iy]==5) and not (i==pairdestx and iy==pairdesty):
                return 1
    #right
    if movearray[ix][iy][1]==1:
        for i in range(max(iy-int(fearparamnew),int(resolution/2)),iy):
            if (array[ix][i]==3 or array[ix][i]==5) and not (ix==pairdestx and i==pairdesty):
                return 1
    #down
    if movearray[ix][iy][2]==1:
        for i in range(max(ix-int(fearparamnew),int(resolution/2)),ix):
            if (array[i][iy]==3 or array[i][iy]==5) and not (i==pairdestx and iy==pairdesty):
                return 1
    #left
    if movearray[ix][iy][3]==1:
        for i in range(iy+1,min(iy+int(fearparamnew)+1,len(array[0])-int(resolution/2))):
            if (array[ix][i]==3 or array[ix][i]==5) and not (ix==pairdestx and i==pairdesty):
                return 1
    #print("fear param is", fearparamnew)
    return 0

#finds the nearest perimiter square following movement directions
def nearestedge(array,ix,iy):
    edge=onedge(ix,iy)
    if edge[0]==1 and ((ix+int(resolution/2)+1)%resolution==0 and (iy+int(resolution/2)+1)%resolution==0):
        n=1
        m=1
    elif edge[0]==1:
        n=0
        m=1
    else:
        n=0 
        m=0
    for i in range(n,max(len(array),len(array[0]))):
        #up
        if (m==1 and movearray[ix][iy][0]==1) or m==0:
            if ix-i>=int(resolution/2):
                edge=onedge(ix-i,iy)
                if edge[0]==1:
                    return(edge[1],edge[2])
        #right
        if (m==1 and movearray[ix][iy][1]==1) or m==0:
            if iy+i<=len(array[0])-int(resolution/2)-1:
                edge=onedge(ix,iy+i)
                if edge[0]==1:
                    return(edge[1],edge[2])
        #down
        if (m==1 and movearray[ix][iy][2]==1) or m==0:
            if ix+i<=len(array)-int(resolution/2)-1:
                edge=onedge(ix+i,iy)
                if edge[0]==1:
                    return(edge[1],edge[2])
        #left
        if (m==1 and movearray[ix][iy][3]==1) or m==0:
            if iy-i>=int(resolution/2):
                edge=onedge(ix,iy-i)
                if edge[0]==1:
                    return(edge[1],edge[2])

#finds if an ion is on the outer perimeter lane
def onedge(ix,iy):
    if ix==int(resolution/2) or ix==len(array)-int(resolution/2)-1 or iy==int(resolution/2) or iy==len(array[0])-int(resolution/2)-1:
        if not (ix<int(resolution/2) or ix>len(array)-int(resolution/2)-1 or  iy<int(resolution/2) or iy>len(array[0])-int(resolution/2)-1):
            return(1,ix,iy)
    return(0,0,0)

#finds nearest perimiter square ignoring movement directions
def nearestedge2(array,ix,iy):
    edge=onedge(ix,iy)
    #print("edge is", edge)
    if edge[0]==1:
        return(edge[1],edge[2])
    for i in range(1,max(len(array),len(array[0]))):
        #up
        if ix-i>=int(resolution/2):
            edge=onedge(ix-i,iy)
            if edge[0]==1:
                return(edge[1],edge[2])
        #right
        if iy+i<=len(array[0])-int(resolution/2)-1:
            edge=onedge(ix,iy+i)
            if edge[0]==1:
                return(edge[1],edge[2])
        #down
        if ix+i<=len(array)-int(resolution/2)-1:
            edge=onedge(ix+i,iy)
            if edge[0]==1:
                return(edge[1],edge[2])
        #left
        if iy-i>=int(resolution/2):
            edge=onedge(ix,iy-i)
            if edge[0]==1:
                return(edge[1],edge[2])
            
#nearest x-junction centre ignoring movementarray
def nearestintnode(array,ix,iy):
    for i in range(0,resolution):
        #up
        if ix-i>0:
            if (ix-i+int(resolution/2)+1)%resolution==0 and (iy+int(resolution/2)+1)%resolution==0 and ix>int(resolution/2):
                return(ix-i,iy)
        #right
        if iy+i<len(array[0]):
            if (iy+i+int(resolution/2)+1)%resolution==0 and (ix+int(resolution/2)+1)%resolution==0 and iy<len(array[0])-int(resolution/2)-1:
                return(ix,iy+i)
        #down
        if ix+i<len(array[0]):
            if (ix+i+int(resolution/2)+1)%resolution==0 and (iy+int(resolution/2)+1)%resolution==0 and ix<len(array)-int(resolution/2)-1 :
                return(ix+i,iy)
        #left
        if iy-i>0:
            if (iy-i+int(resolution/2)+1)%resolution==0 and (ix+int(resolution/2)+1)%resolution==0 and iy>int(resolution/2):
                return(ix,iy-i)
            
#nearest x-junction centre following movementarray
def nearestnode(x,y):
    #print(movearray[x][y])
    if (x+(int(resolution/2)+1))%resolution==0 and (y+(int(resolution/2)+1))%resolution==0:
        return(x,y)
    if (x+(int(resolution/2)+1))%resolution==0:
        if movearray[x][y][1]==1:
            for i in range(1,resolution):
                #print(y+i)
                if y-i<0:
                    break
                if (y-i+(int(resolution/2)+1))%resolution==0:
                    return(x,y-i)
        if movearray[x][y][3]==1:
            for i in range(1,resolution):
                if y+i>len(array[0]):
                    break
                if (y+i+(int(resolution/2)+1))%resolution==0:
                    return(x,y+i)
    if (y+(int(resolution/2)+1))%resolution==0:
        if movearray[x][y][0]==1:
            for i in range(1,resolution):
                if x+i>len(array):
                    break
                if (x+i+(int(resolution/2)+1))%resolution==0:
                    return(x+i,y)
        if movearray[x][y][2]==1:
            for i in range(1,resolution):
                if x-i<0:
                    break
                if (x-i+(int(resolution/2)+1))%resolution==0:
                    return(x-i,y)
    else:
        #print("nearest node broke for", x,y)           
        return (int(resolution/2),int(resolution/2))
    
def perimoccupied(array,ix,iy,direction,pairdestx,pairdesty):
    if direction==0:
        for i in range(1,int(resolution/2)+1):
            if (array[ix-i][iy]==3 or array[ix-i][iy]==5) and not (ix-i==pairdestx and iy==pairdesty):
                return 1
    elif direction==1:
        for i in range(1,int(resolution/2)+1):
            if (array[ix][iy+i]==3 or array[ix][iy+i]==5) and not (ix==pairdestx and iy+i==pairdesty):
                return 1
    elif direction==2:
        for i in range(1,int(resolution/2)+1):
            if (array[ix+i][iy]==3 or array[ix+i][iy]==5) and not (ix+i==pairdestx and iy==pairdesty):
                return 1
    elif direction==3:
        for i in range(1,int(resolution/2)+1):
            if (array[ix][iy-i]==3 or array[ix][iy-i]==5) and not (ix==pairdestx and iy-i==pairdesty):
                return 1
    return 0
 
#finds exact minmium distance between two points ignoring lane system
def dist(x1,y1,x2,y2):
    node1=[]
    node2=[]
    dist1=9999
    dist2=99999
    combination=0
    for i in range (0,resolution):
        if (x1+(int(resolution/2)+1)+i)%resolution==0 and (y1+(int(resolution/2)+1))%resolution==0 and x1+i<=len(array):
            node1.append([x1+i,y1])
        if (x1+(int(resolution/2)+1)-i)%resolution==0 and (y1+(int(resolution/2)+1))%resolution==0 and x1-i>=0:
            node1.append([x1-i,y1])
        if (x1+(int(resolution/2)+1))%resolution==0 and (y1+(int(resolution/2)+1)-i)%resolution==0 and y1-i>=0:
            node1.append([x1,y1-i])
        if (x1+(int(resolution/2)+1))%resolution==0 and (y1+(int(resolution/2)+1)+i)%resolution==0 and y1+i<=len(array[0]):
            node1.append([x1,y1+i])
           
        
        if (x2+(int(resolution/2)+1)+i)%resolution==0 and (y2+(int(resolution/2)+1))%resolution==0 and x2+i<=len(array):
            node2.append([x2+i,y2])
        if (x2+(int(resolution/2)+1)-i)%resolution==0 and (y2+(int(resolution/2)+1))%resolution==0 and x2-i>=0:
            node2.append([x2-i,y2])
        if (x2+(int(resolution/2)+1))%resolution==0 and (y2+(int(resolution/2)+1)-i)%resolution==0 and y2-i>=0:
            node2.append([x2,y2-i])
        if (x2+(int(resolution/2)+1))%resolution==0 and (y2+(int(resolution/2)+1)+i)%resolution==0 and y2+i<=len(array[0]):
            node2.append([x2,y2+i])
    for i in range (0,len(node1)):
        for j in range (0,len(node2)):
            dist1=manhattandistance(node1[i][0],node1[i][1],node2[j][0],node2[j][1])+manhattandistance(x1,y1,node1[i][0],node1[i][1])+manhattandistance(x2,y2,node2[j][0],node2[j][1])
            if dist1<dist2:
                combination=[node1[i][0],node1[i][1],node2[j][0],node2[j][1]]
                dist2=dist1
    #print("combination is", combination)
    return combination,dist2

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

#a* path finding algorithm to find shortest route following lane system - used but not neccessary with current version of code could be replaced
def astar(maze, movearray, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    lengthx=len(maze)
    lengthy=len(maze[0])
    # Initialize both open and closed list
    open_list = []
    closed_list = []
    # Add the start node
    open_list.append(start_node)
    start = time.time()
    # Loop until you find the end
    while len(open_list) > 0 and time.time()-start<0.3:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        #print(current_node.position[0],current_node.position[1])

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        
        posmove=[]
        
    
            
        if movearray[current_node.position[0]][current_node.position[1]][0]==1 and current_node.position[0]-resolution>0 :
            posmove.append((-resolution,0))
        if movearray[current_node.position[0]][current_node.position[1]][1]==1 and current_node.position[1]+resolution<lengthy:
            posmove.append((0,resolution))
        if movearray[current_node.position[0]][current_node.position[1]][2]==1 and current_node.position[0]+resolution<lengthx:
            posmove.append((resolution,0))
        if movearray[current_node.position[0]][current_node.position[1]][3]==1 and current_node.position[1]-resolution>0:
            posmove.append((0,-resolution))
        #print ("posmove is", posmove)
        for new_position in posmove: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            #if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze) - 1) or node_position[1] < 0:
                #continue
            
            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] ==0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            #child.h=0
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
            #print(child.position[0],child.position[1])


def main(array,movearray,start,end):
    try:
        path = astar(array,movearray, start, end)
        len(path)
        return path        
    except TypeError:
        print("path finder broke", start, end)
        return ((int(resolution/2),int(resolution/2)),(int(resolution/2),int(resolution/2)))
    
    
    
#calculate distances
def bestMove(ax,ay,destx,desty,move):
    distanceList=[100000,100000,100000,100000]
    bestMove=4
    for k in range (0,4):
        if k==0 and move[k]==1:
            distanceList[k]=distance(ax-1,ay,destx,desty)
        elif k==1 and move[k]==1:
            distanceList[k]=distance(ax,ay+1,destx,desty)
        elif k==2 and move[k]==1:
            distanceList[k]=distance(ax+1,ay,destx,desty)
        elif k==3 and move[k]==1:
            distanceList[k]=distance(ax,ay-1,destx,desty)    
    if ax==destx and ay==desty:
        for k in range(0,35):
            rand=random.randint(0,3)
            if move[rand]==1:
                return rand        
    else:
        return distanceList.index(min(distanceList))
    
    
def remove_values_from_list(the_list, val):
    for i in range(the_list.count(val)):
        the_list.remove(val)


###################################################################################################
        
#starting device size-1
numberx=3
numbery=3

for j in range(0,1):
    
    #increasing device size 
    numberx+=1
    numbery+=1
    
    #number of iterations per device size
    iterations=10
    
    #pause per movement
    sleep=0.5

    #number of ions initialised per x-junction
    ion=2
    
    #what percentage of ions participate in a gate
    gatedensity=1
    
    #resolution of model
    resolution=9 

    #how many squares ions maintain between themselves
    fearparam=2

    #total number of ions
    ions=ion*numberx*numbery

    #data extraction lists
    movementcount=[]
    lowerboundcount=[]
    xmovementcount1=[]
    xmovementcount2=[]
    xmovementcount=[]
    for i in range(0,iterations*ions):
        xmovementcount1.append(0)
        xmovementcount.append(0)
    for i in range(0,iterations*ions):
        xmovementcount2.append(0)

        
    #run this device size for this many iterations    
    for f in range(0,iterations):

        #custom ion pairing not set up "manual", so always =0 for now
        manualmode=0
        
        
        perimcheck=1
        numgates=numberx*numbery
        hh=int(resolution/2)

        #total number ion pairs
        totpos2gates=np.ceil(ions/2)     
        
        #total pairs to go to gates
        pairs2gates=int(np.ceil(totpos2gates*gatedensity))

        #number of required rounds based on pairs to gates and available gates
        rounds=int(np.ceil(pairs2gates/(numgates)))
       
        #calculating the pairs per round and the remainder in the final round
        numrounds=[0,0]
        pairsperround=[]
        pairsremaining=pairs2gates
        for i in range(0,rounds):
            pairsperround.append(min(numgates,pairsremaining))
            pairsremaining-=min(numgates,pairsremaining)


        #display paramaters
        # This sets the WIDTH and HEIGHT of each grid location
        WIDTH = int(200/(((numberx*numbery)**0.5)*resolution**0.6))
        HEIGHT = int(200/(((numberx*numbery)**0.5)*resolution**0.6))
        # This sets the margin between each cell
        MARGIN = 1

        #create the position and movement array
        array=XJunction(numberx,numbery,resolution)
        movearray=createMoveArray(array)
        array=XJunction(numberx,numbery,resolution)

        #list of ion positions
        a=[]
        #list of gate positions
        b=[]
        # ion database
        c=[]
        dest=[]
        dest2=[]

        #create gates b is a list of gate positions
        for i in range (0,numbery):
                for j in range (0,numberx):
                    if i==0:
                        array[j*resolution+(int(resolution/2))][0]=7
                        b.append(j*resolution+(int(resolution/2)))
                        b.append(0)
                    elif i==numbery-1:
                        array[j*resolution+(int(resolution/2))][i*resolution+resolution-1]=7
                        b.append(j*resolution+(int(resolution/2)))
                        b.append(i*resolution+resolution-1)
                    elif j==0:
                        array[j*resolution][i*resolution+(int(resolution/2))]=7
                        b.append(j*resolution)
                        b.append(i*resolution+(int(resolution/2)))
                    elif j==numberx-1:
                        array[j*resolution+resolution-1][i*resolution+(int(resolution/2))]=7
                        b.append(j*resolution+resolution-1)
                        b.append(i*resolution+(int(resolution/2)))
                    elif (i) % 2 == 0:
                        array[j*resolution][i*resolution+(int(resolution/2))]=7
                        b.append(j*resolution)

                        b.append(i*resolution+(int(resolution/2)))
                    else:
                        array[j*resolution+resolution-1][i*resolution+(int(resolution/2))]=7
                        b.append(j*resolution+resolution-1)
                        b.append(i*resolution+(int(resolution/2)))

        #Write to file
        f1= open("init.txt","w+")
        for i in range(0,len(array)):
                for j in range(0,len(array[0])):

                    f1.write(str(array[i][j]) + "\t")
                f1.write("\n")
        f1.close() 

        for i in range (0,ions):
            c.append([0,0,0,0])
            dest.append([0,0])
            dest2.append([0,0])

        #create ions, a, is a list of ion locations
        loop=0
        ioncount=0
        while ioncount<ions:
            loop+=1
            for i in range (0,numberx):
                if ioncount>=ions:
                    break
                for j in range (0,numbery):

                    if j==0 and i==numberx-1:
                        array[i*resolution-loop+(int(resolution/2))][j*resolution+(int(resolution/2))]=3
                        a.append(i*resolution-loop+(int(resolution/2)))
                        a.append(j*resolution+(int(resolution/2)))
                        ioncount+=1
                        if ioncount>=ions:
                            break
                    elif j==0:
                        array[i*resolution+loop+(int(resolution/2))][j*resolution+(int(resolution/2))]=3
                        a.append(i*resolution+loop+(int(resolution/2)))
                        a.append(j*resolution+(int(resolution/2)))
                        ioncount+=1
                        if ioncount>=ions:
                            break

                    else:
                        array[i*resolution+(int(resolution/2))][j*resolution-loop+(int(resolution/2))]=3
                        a.append(i*resolution+(int(resolution/2)))
                        a.append(j*resolution-loop+(int(resolution/2)))
                        ioncount+=1
                        if ioncount>=ions:
                            break

        # Initialize pygame
        pygame.init()
        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [(WIDTH+MARGIN)*len(array[0]), (HEIGHT+MARGIN)*len(array)]
        screen = pygame.display.set_mode(WINDOW_SIZE)
        # Set title of screen
        pygame.display.set_caption("Array Backed Grid")
        # Loop until the user clicks the close button.
        done = False
        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()
        f2= open("ion.txt","w+")
        done=0
        destinize=1
        count=0
        secondcount=0
        lowerbound=0
        
        #Begin
        while done==0:
            
            movementcheck=0
            count+=1
            #is reset when multiple rounds are needed
            secondcount+=1
            
            #extract positions to a file for 3D rendering
            for i in range(0,ions):  
                f2.write(str(count) + "\t" + str(i) + "\t" + str(a[2*i]) + "\t" + str(a[2*i+1]) + "\t" + str(c[i][3]) + "\n")


            
            #pygame stuff
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True  # Flag that we are done so we exit this loop
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # User clicks the mouse. Get the position
                    pygame.quit()
                    f2.close() 
                    output=0
                    for i in range(0,ions):
                        if c[i][2]==1:
                            output+=distance(a[2*i],a[2*i+1],a[2*c[i][0]],a[2*c[i][0]+1])
                    #print("DISTANCE", ouratput)
                    break                
            # Set the screen background
            screen.fill(BLACK)
            time.sleep(sleep)


            #moveions
            for j in range (0,ions):
                i=2*j
                
                if (a[i]+int(resolution/2)+1)%resolution!=0 or (a[i+1]+int(resolution/2)+1)%resolution!=0:
                    oncentre=0

                if (a[i]+int(resolution/2)+1)%resolution==0 and (a[i+1]+int(resolution/2)+1)%resolution==0:
                    oncentre=1

                if manualmode==0 and numrounds[0]==0 and numrounds[1]==0:
                    destinize=0
                    
                    #pairing qubits
                    if float(pairs2gates)>ions/2:
                        single=1
                    else:
                        single=0
                    choose=random.sample(range(0, ions-single), 2*(pairs2gates-single))
                    
                    for k in range (0,pairs2gates-single):
                        c[choose[2*k]]=[choose[2*k+1],0,1,0]
                        c[choose[2*k+1]]=[choose[2*k],0,1,0]
                    if single==1:
                        c[ions-1]=[ions-1,0,3,0]

                #assigning gates as destinations
                if manualmode==0 and numrounds[1]==0:
                    numrounds[1]=1
                    secondcount=1
                    usedgate=[]
                    assigned=0
                    minshuttlelist=[]
                    if numrounds[0]>=1:
                        for k in range (0,ions):
                            if c[k][2]==2 and not (a[2*k]==0 and a[2*k+1]==0):
                                c[k][2]=4
                                c[c[k][0]][2]=4
                            
                    for k in range (0,ions):
                        if assigned>=(pairsperround[numrounds[0]]):
                            break
                        totDist1=100
                        totDist2=200
                        if c[k][2]==1:
                            for h in range(0,numberx*numbery):
                                if h in usedgate:
                                    y=1
                                else:
                                    node1=nearestnode(a[2*k],a[2*k+1]) 
                                    node2=nearestnode(a[2*c[k][0]],a[2*c[k][0]+1])
                                    node3=nearestnode(b[2*h],b[2*h+1])
                                    totDist1=max(np.size(main(array,movearray,(node1[0],node1[1]),(node3[0],node3[1]))),np.size(main(array,movearray,(node2[0],node2[1]),(node3[0],node3[1]))))
                                    #totDist1=distance((a[2*k]+a[2*c[k][0]])/2,(a[2*k+1]+a[2*c[k][0]+1])/2,b[2*h],b[2*h+1])
                                    if totDist1<totDist2:
                                        gate=h
                                        totDist2=totDist1
                            c[k][1]=gate
                            c[k][2]=2
                            c[c[k][0]][1]=gate
                            c[c[k][0]][2]=2
                            usedgate.append(gate)
                            assigned+=1
                            #print("sending ion", c[k], "and", c[c[k][0]], "to gate", gate)
                    #lowerbound calculation
                    for k in range(0,ions):
                         if c[k][2]==2 and not (a[2*k]==0 and a[2*k+1]==0):
                            minshuttlelist.append(dist(a[2*k],a[2*k+1],b[2*c[k][1]],b[2*c[k][1]+1])[1])
                    lowerbound+=max(minshuttlelist)
                    #print("lowerbound is", lowerbound/resolution)

                #set destination
                if c[j][2]==0:
                    c[j]=[j,0,0,0]
                gatedest=c[j][1]
                pair=c[j][0]
                pairdestx=a[2*pair]
                pairdesty=a[2*pair+1]
                if c[j][2]==2 and (b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1):
                    dest[j]=([b[(2*gatedest)],b[2*gatedest+1]])
                    
                #Unique waiting ion at centre of x-junction behaviour, changing destination when needed
                elif c[j][2]==2:
                    outedge=nearestintnode(array,b[(2*gatedest)],b[(2*gatedest)+1])

                    if movearray[outedge[0]][outedge[1]][0]==1 and secondcount==1:    
                        dest[j]=([outedge[0]+1,outedge[1]])
                        #print(j, dest[j])
                    elif secondcount==1:
                        dest[j]=([outedge[0]-1,outedge[1]])
                        #print(j, dest[j])
                    if a[i]==dest[j][0] and a[i+1]==dest[j][1] and need2move(array,a[i],a[i+1],movearray,pairdestx,pairdesty,1,c[j][2],dest[j][0],dest[j][1])==1:
                        #print("2 ion", j, movearray[outedge[0]][outedge[1]])
                        if movearray[outedge[0]][outedge[1]]==[1,1,0,0]:
                            if (a[i+1]+int(resolution/2)+1)%resolution==0:
                                dest[j][0]-=1
                                dest[j][1]-=1
                            else:
                                dest[j][0]+=1
                                dest[j][1]+=1
                        if movearray[outedge[0]][outedge[1]]==[1,0,0,1]:
                            if (a[i+1]+int(resolution/2)+1)%resolution==0:
                                dest[j][0]-=1
                                dest[j][1]+=1
                            else:
                                dest[j][0]+=1
                                dest[j][1]-=1
                        if movearray[outedge[0]][outedge[1]]==[0,1,1,0]:
                            if (a[i+1]+int(resolution/2)+1)%resolution==0:
                                dest[j][0]+=1
                                dest[j][1]-=1
                            else:
                                dest[j][0]-=1
                                dest[j][1]+=1
                        if movearray[outedge[0]][outedge[1]]==[0,0,1,1]:
                            if (a[i+1]+int(resolution/2)+1)%resolution==0:
                                dest[j][0]+=1
                                dest[j][1]+=1
                            else:
                                dest[j][0]-=1
                                dest[j][1]-=1                 
                #elif j<pair and perimcheck<0:
                    #chase pair while perimiters not filled
                    #dest[j]=(pairdestx,pairdesty)

                else: 

                    if c[j][2]==4 and (not (a[i]==0 and a[i+1]==0)) and (a[i]<int(resolution/2) or a[i]>(len(array)-int(resolution/2)-1) or a[i+1]<int(resolution/2) or a[i+1]>(len(array[0])-int(resolution/2)-1)):
                        outedge=nearestedge2(array,a[i],a[i+1])
                        dest[j]=([outedge[0],outedge[1]])
                    elif c[j][2]!=2 and (not (a[i]==0 and a[i+1]==0)) and perimcheck==1 and j<pair:
                        dest[j]=(pairdestx,pairdesty)
                    else:
                        dest[j]=([a[i],a[i+1]])


                if c[j][2]==2 and (a[i]+int(resolution/2)+1)%resolution==0 and (a[i+1]+int(resolution/2)+1)%resolution==0 and distance(a[i],a[i+1],dest[j][0],dest[j][1])>resolution:   
                        node=nearestnode(dest[j][0],dest[j][1]) 
                        nodemove=main(array,movearray,(a[i],a[i+1]),(node[0],node[1]))
                        if np.size(nodemove)>=4:
                            dest2[j][0]=nodemove[1][0]
                            dest2[j][1]=nodemove[1][1]
                        else:
                            dest2[j][0]=nodemove[0][0]
                            dest2[j][1]=nodemove[0][1]
                else:
                    dest2[j][0]=dest[j][0]
                    dest2[j][1]=dest[j][1]

                #Find valid move
                move=[0,0,0,0]
                pmove=[0,0,0,0]

                #decide which moves are available - main bulk of routing logic here, up,right,down,left
                #up direction
                if a[i]-1==pairdestx and a[i+1]==pairdesty:
                    pmove[0]=1
                elif a[i]==(int(resolution/2)) and perimoccupied(array,a[i],a[i+1],0,pairdestx,pairdesty)==1:
                    move[0]=0
                elif a[i]==(int(resolution/2)) and distance(a[i]-(int(resolution/2)),a[i+1],dest2[j][0],dest2[j][1])!=0:
                    move[0]=0
                elif a[i]-1>=0 and a[i]==dest2[j][0] and a[i+1]==dest2[j][1] and movearray[a[i]][a[i+1]][0]==1 and (array[a[i]-1][a[i+1]]==1 or array[a[i]-1][a[i+1]]==7):
                    if need2move(array,a[i],a[i+1],movearray,pairdestx,pairdesty,fearparam, c[j][2],dest[j][0],dest[j][1])==1 and a[i]!=len(array)-1:
                        move[0]=1
                    else:
                        move[0]=0
                elif a[i]-1>=0 and (array[a[i]-1][a[i+1]]==1 or array[a[i]-1][a[i+1]]==7)  and movearray[a[i]][a[i+1]][0]==1:
                    move[0]=1
                elif a[i]-1>=0 and (array[a[i]-1][a[i+1]]==1 or array[a[i]-1][a[i+1]]==7)  and a[i]-1==dest2[j][0] and a[i+1]==dest2[j][1]:
                    move[0]=1
                #right
                if a[i]==pairdestx and a[i+1]+1==pairdesty:
                    pmove[1]=1
                elif a[i+1]==(len(array[0])-1-int(resolution/2)) and perimoccupied(array,a[i],a[i+1],1,pairdestx,pairdesty)==1:
                    move[1]=0
                elif a[i+1]==(len(array[0])-1-int(resolution/2)) and distance(a[i],a[i+1]+(int(resolution/2)),dest2[j][0],dest2[j][1])!=0:
                    move[1]=0
                elif a[i+1]+1<len(array[0]) and a[i]==dest2[j][0] and a[i+1]==dest2[j][1] and movearray[a[i]][a[i+1]][1]==1 and (array[a[i]][a[i+1]+1]==1 or array[a[i]][a[i+1]+1]==7):
                    if need2move(array,a[i],a[i+1],movearray,pairdestx,pairdesty,fearparam,c[j][2],dest[j][0],dest[j][1])==1 and a[i+1]!=0:
                        move[1]=1
                    else:
                        move[1]=0
                elif a[i+1]+1<len(array[0]) and (array[a[i]][a[i+1]+1]==1 or array[a[i]][a[i+1]+1]==7) and movearray[a[i]][a[i+1]][1]==1:
                    move[1]=1
                elif a[i+1]+1<len(array[0]) and (array[a[i]][a[i+1]+1]==1 or array[a[i]][a[i+1]+1]==7) and a[i]==dest2[j][0] and a[i+1]+1==dest2[j][1]:
                    move[1]=1
                #down
                if a[i]+1==pairdestx and a[i+1]==pairdesty:
                    pmove[2]=1
                elif a[i]==(len(array)-1-int(resolution/2)) and perimoccupied(array,a[i],a[i+1],2,pairdestx,pairdesty)==1:
                    move[2]=0
                elif a[i]==(len(array)-1-int(resolution/2)) and distance(a[i]+(int(resolution/2)),a[i+1],dest2[j][0],dest2[j][1])!=0:
                    move[2]=0
                elif a[i]+1<len(array) and a[i]==dest2[j][0] and a[i+1]==dest2[j][1] and movearray[a[i]][a[i+1]][2]==1 and (array[a[i]+1][a[i+1]]==1 or array[a[i]+1][a[i+1]]==7):
                    if need2move(array,a[i],a[i+1],movearray,pairdestx,pairdesty,fearparam, c[j][2],dest[j][0],dest[j][1])==1 and a[i]!=0:
                        move[2]=1
                    else:
                        move[2]=0 
                elif a[i]+1<len(array) and (array[a[i]+1][a[i+1]]==1 or array[a[i]+1][a[i+1]]==7) and movearray[a[i]][a[i+1]][2]==1:
                    move[2]=1
                elif a[i]+1<len(array) and (array[a[i]+1][a[i+1]]==1 or array[a[i]+1][a[i+1]]==7) and a[i]+1==dest2[j][0] and a[i+1]==dest2[j][1]:
                    move[2]=1
                #left
                if a[i]==pairdestx and a[i+1]-1==pairdesty:
                    pmove[3]=1
                elif a[i+1]==int(resolution/2) and perimoccupied(array,a[i],a[i+1],3,pairdestx,pairdesty)==1:
                    move[3]=0
                elif a[i+1]==int(resolution/2) and distance(a[i],a[i+1]-(int(resolution/2)),dest2[j][0],dest2[j][1])!=0:
                    move[3]=0
                elif a[i+1]-1>=0 and  a[i]==dest2[j][0] and a[i+1]==dest2[j][1] and movearray[a[i]][a[i+1]][3]==1 and (array[a[i]][a[i+1]-1]==1 or array[a[i]][a[i+1]-1]==7):
                    if need2move(array,a[i],a[i+1],movearray,pairdestx,pairdesty,fearparam, c[j][2],dest[j][0],dest[j][1])==1 and a[i+1]!=len(array[0])-1:
                        move[3]=1
                    else:
                        move[3]=0 
                elif a[i+1]-1>=0 and (array[a[i]][a[i+1]-1]==1 or array[a[i]][a[i+1]-1]==7) and movearray[a[i]][a[i+1]][3]==1:
                    move[3]=1
                elif a[i+1]-1>=0 and (array[a[i]][a[i+1]-1]==1 or array[a[i]][a[i+1]-1]==7)  and a[i]==dest2[j][0] and a[i+1]-1==dest2[j][1]:
                    move[3]=1

                #move
                if sum(pmove)>0:
                    if pmove[0]==1:
                        array[a[i]][a[i+1]]=1
                        array[a[i]-1][a[i+1]]=3
                        a[i]=a[i]-1
                        a[2*pair]=0
                        a[2*pair+1]=0
                        c[pair][3]=1
                        c[j][3]=1
                    if pmove[1]==1:
                        array[a[i]][a[i+1]]=1
                        array[a[i]][a[i+1]+1]=3
                        a[i+1]=a[i+1]+1
                        a[2*pair]=0
                        a[2*pair+1]=0
                        c[pair][3]=1
                        c[j][3]=1
                    if pmove[2]==1:
                        array[a[i]][a[i+1]]=1
                        array[a[i]+1][a[i+1]]=3
                        a[i]=a[i]+1
                        a[2*pair]=0
                        a[2*pair+1]=0
                        c[pair][3]=1
                        c[j][3]=1
                    if pmove[3]==1:
                        array[a[i]][a[i+1]]=1
                        array[a[i]][a[i+1]-1]=3
                        a[i+1]=a[i+1]-1
                        a[2*pair]=0
                        a[2*pair+1]=0
                        c[pair][3]=1
                        c[j][3]=1

                #choose move that minimises distance from destination      
                elif sum(move)>0:    
                    if 1==2 and c[k][2]==1 and a[i]==dest2[j][0] and a[i+1]==dest2[j][1]:
                        #move towards pair when need 2 move - but maybe bugs all into a corner
                        BESTMove=bestMove(a[i],a[i+1],pairdestx,pairdesty,move)
                    else:
                        BESTMove=bestMove(a[i],a[i+1],dest2[j][0],dest2[j][1],move)
                    if BESTMove==0:
                        array[a[i]][a[i+1]]=1
                        array[a[i]-1][a[i+1]]=3
                        a[i]=a[i]-1

                    elif BESTMove==1:
                        array[a[i]][a[i+1]]=1
                        array[a[i]][a[i+1]+1]=3
                        a[i+1]=a[i+1]+1

                    elif BESTMove==2:
                        array[a[i]][a[i+1]]=1
                        array[a[i]+1][a[i+1]]=3
                        a[i]=a[i]+1

                    elif BESTMove==3:
                        array[a[i]][a[i+1]]=1
                        array[a[i]][a[i+1]-1]=3
                        a[i+1]=a[i+1]-1

                else:
                    movementcheck+=1


                #splitting xmovementcount for ion type inter/exter
                if ((b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1)):
                    xmovementcount2[f*ions+j]=1337
                if not ((b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1)):
                    xmovementcount1[f*ions+j]=1337

                if ((b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1)) and (a[i]+int(resolution/2)+1)%resolution==0 and (a[i+1]+int(resolution/2)+1)%resolution==0 and oncentre==0:
                    xmovementcount1[f*ions+j]+=1

                if  (a[i]+int(resolution/2)+1)%resolution==0 and (a[i+1]+int(resolution/2)+1)%resolution==0 and oncentre==0 and not ((b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1)):
                    xmovementcount2[f*ions+j]+=1

                #if  (a[i]+int(resolution/2)+1)%resolution==0 and (a[i+1]+int(resolution/2)+1)%resolution==0 and oncentre==0:
                    #xmovementcount[f*ions+j]+=1


            if lowerbound/count<0.05:
                print("IT BROKE")
                break

            if numrounds[1]==2 and movementcheck==ions and numrounds[0]==rounds-1:
                #print(count/resolution,lowerbound/resolution)
                movementcount.append(count/resolution)
                lowerboundcount.append(lowerbound/resolution)
                break
            if numrounds[1]==2 and movementcheck==ions and numrounds[0]<rounds-1:
                numrounds[0]+=1
                numrounds[1]=0
                #print(numrounds)

            if numrounds[1]==1 and movementcheck==ions:
                for z in range(0,ions):
                    gatedest=c[z][1]
                    if c[z][2]==2 and not (b[(2*gatedest)]==0 or b[(2*gatedest)]==len(array)-1 or b[(2*gatedest+1)]==0 or b[(2*gatedest)+1]==len(array[0])-1):
                        dest[z]=([b[(2*gatedest)],b[2*gatedest+1]])
                numrounds[1]=2


            # Draw the grid    
            for h in range(0,numberx*numbery):
                if array[b[2*h]][b[2*h+1]]==1:
                    array[b[2*h]][b[2*h+1]]=7
            for h in range(0,ions):
                if c[h][3]==1:
                    array[a[2*h]][a[2*h+1]]=5
                else:
                    array[a[2*h]][a[2*h+1]]=3


            #this isn't checking whether perimgate is even assigned
            if count%(hh)==0:
                perimcheck=1
                for z in range(0,numberx*numbery):
                    if (b[(2*z)]==0 or b[(2*z)]==len(array)-1 or b[(2*z+1)]==0 or b[(2*z)+1]==len(array[0])-1):
                        #print("value of array at edge gatezone", array[b[2*z]][b[2*z+1]])
                        if array[b[2*z]][b[2*z+1]]==(5 or 3):
                            perimcheck=-1
                        else:
                            perimcheck=1
                            break
                #print(perimcheck)

            for row in range(len(array)):
                for column in range(len(array[0])):
                    color = WHITE
                    if array[row][column] == 1:
                        color = GREEN
                    if array[row][column] == 3:
                        color = RED
                    if array[row][column] == 5:
                        color = ORANGE
                    if array[row][column] == 7:
                        color = BLUE
                    pygame.draw.rect(screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])

            # Limit to 60 frames per second
            clock.tick(60)

            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()


        # Be IDLE friendly. If you forget this line, the program will 'hang'
        # on exit.
        pygame.quit()
        f2.close()
        #print(xmovementcount)
    remove_values_from_list(xmovementcount1,1337)
    remove_values_from_list(xmovementcount2,1337)
    #print(xmovementcount1)
    #print(xmovementcount2)


    print(ion, numberx, ions, np.mean(movementcount), "+/-", np.std(movementcount), np.mean(lowerboundcount),"+/-", np.std(lowerboundcount),np.mean(xmovementcount1),"+/-", np.std(xmovementcount1),np.mean(xmovementcount2),"+/-", np.std(xmovementcount2))
    #print(ion, numberx, ions, np.mean(movementcount), "+/-", np.std(movementcount), np.mean(lowerboundcount),"+/-", np.std(lowerboundcount),np.mean(xmovementcount),"+/-", np.std(xmovementcount))