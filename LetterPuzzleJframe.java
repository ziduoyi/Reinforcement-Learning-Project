
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import javax.imageio.*;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

 // JOptionPane for beginning
public class LetterPuzzleJframe
{
	public static void main(String...args) throws IOException
	{
		JFrame j = new JFrame();  //JFrame is the window; window is a depricated class
        int wid = (new OptionPaneExample()).width;
		MyPanelb m = new MyPanelb(wid);
		j.setSize(m.getSize());
		j.add(m); //adds the panel to the frame so that the picture will be drawn
			      //use setContentPane() sometimes works better then just add b/c of greater efficiency.
        j.addMouseListener(m);
		j.setVisible(true); //allows the frame to be shown.
        //m.solve();
		j.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //makes the dialog box exit when you click the "x" button.
	}

}

class MyPanelb extends JPanel implements MouseListener
{
	Rectangle grid[][];
	boolean scramble[];
	String matrix[][], empty;
	Random rnd;
	int blankR, blankC, playLevel, side, width;
    Map<Integer, BufferedImage> map = new HashMap<>();
	MyPanelb(int wid) throws IOException
	{
        side = wid;//change later
		setSize(2000, 1500);
		grid = new Rectangle[side+2][side+2];
        BufferedImage  img = ImageIO.read(new File("C:\\Users\\ziduo\\Python\\New Python Stuff\\Lawn Mower\\image1.jpg")); //needs change
        width = 600/side;
        for(int i=0; i<side; i++)
            for(int j=0; j<side; j++){
                BufferedImage temp = img.getSubimage((i)*width, (j)*width, width, width);
                map.put(i*side + j, temp);
            }

		
        empty = Character.toString('A'+side*side-1);
		for(int i=0; i<=side+1; i++)
			for(int j=0; j<=side+1; j++)
				grid[i][j] = new Rectangle(100+(i-1)*width, 100+(j-1)*width, width, width);
        
		boolean solve = false;
        while(!solve){
            matrix = new String[side+2][side+2];
            scramble = new boolean[side*side+1];
            rnd = new Random();
            
            for (int r = 0; r <= side+1; r++)
                for (int c = 0; c <= side+1; c++)
                    matrix[r][c] = "#";	
            
            for (int r = 1; r <= side; r++)
                for (int c = 1; c <= side; c++)
                {
                    matrix[r][c] = getLetter();
                    if (matrix[r][c].equals(empty))
                    {
                        blankR = r;
                        blankC = c;
                    }
                }
            solve = isSolvable();
        }
		setVisible(true); //it's like calling the repaint method.
	
	
	}
	
	public String getLetter()
	{
		String letter = "";
		boolean Done = false;
		while(!Done)
		{
			int rndNum = rnd.nextInt(side*side) + 1;
			if (scramble[rndNum] == false)
			{
				letter = String.valueOf((char) (rndNum+64));
				scramble[rndNum] = true;
				Done = true;
			}
		}
		return letter;		
	}
	
	public void paintComponent(Graphics g)
	{
        drawGrid(g);
        for(int i=1; i<=side; i++)
            for(int j=1; j<=side; j++){
                //drawLetter(g,matrix[i][j],100+(i-1)*width,100+(j-1)*width);
                g.drawImage(map.get(matrix[i][j].toCharArray()[0]-'A'), 100+(i-1)*width, 100+(j-1)*width, width, width, null);
            }
		
	}
	
	public void drawGrid(Graphics g)
	{
		g.drawRect(100,100,600,600);
        for(int i=1; i<side; i++){
            g.drawLine(100,100+width*i,700,100+width*i);
            g.drawLine(100+width*i,100,100+width*i,700);
        }
	}
	
	
	public void drawLetter(Graphics g, String letter, int x, int y)
	{
		int offSetX = y+(width)/4;
		int offSetY = x+(width)-(width)/4;
		g.setFont(new Font("Arial",Font.BOLD,width*2/3));
		if (letter.equals(empty))
		{
			g.setColor(Color.white);
			g.fillRect(y+1,x+1,width-2,width-2);
		}
		else
		{
            g.setColor(Color.lightGray);
			g.fillRect(y+1,x+1,width-2,width-2);
			g.setColor(Color.black);
			g.drawString(letter,offSetX,offSetY);			
		}
	}
	

	public void mousePressed(MouseEvent e){}
	public void mouseReleased(MouseEvent e){}
	public void mouseEntered(MouseEvent e){}
	public void mouseExited(MouseEvent e){}
	
	public void mouseClicked(MouseEvent e)
	{
		int x = e.getX();
		int y = e.getY();
        for(int i = 1; i <= side; i++){
            for(int j = 1; j <= side; j++){
                if(grid[i][j].contains(x, y) && okSquare(i, j)){ //used to be inside
                    swap(i, j);
                    return;
                }
            }
        }				
		//return true;
	}
    int[][] direct = {{0, 1},{-1, 0},{1, 0},{0, -1}};
    public void solve(){
        System.out.println(isSolvable());
        fullSolver(getArr());
        int[][] temp = new int[side][side];
        int[] empty = new int[2];
        for(int i=0; i<side; i++)
            for(int j=0; j<side; j++){
                temp[i][j] = matrix[i+1][j+1].toCharArray()[0]-'A';
                if(temp[i][j] == side*side - 1){
                    empty[0] = i;
                    empty[1] = j;
                }
            }
        Map<String, Integer> space = new HashMap<>();
        LinkedList<int[][]> states = new LinkedList<>();
        LinkedList<int[]> holes = new LinkedList<>();
        LinkedList<int[]> axis = new LinkedList<>();
        axis.add(new int[] {Math.max(0, side-3), Math.max(0, side-3)});
        states.add(temp);
        space.put(convert(temp), -1);
        holes.add(empty);
        String finish = "";
        int min = 0;
        while(!states.isEmpty()){
            int[][] state = states.removeFirst();
            boolean done = true;
            int[] border = axis.removeFirst();
            int[] hole = holes.removeFirst();
            if(Math.max(border[0], border[1])<min)
                continue;
            if(Math.min(border[0], border[1])>min)
                System.out.println(Math.min(border[0], border[1]));
            min = Math.max(min, Math.min(border[0], border[1]));
            for(int i=0; i<side; i++){
                for(int j=0; j<side; j++)
                    if(state[i][j] != side * (i) + j){
                        done = false;
                        break;
                    }
                if(!done)
                    break;
            }
            if(done){
                finish = convert(state);
                break;
            }
            
            for(int i=0; i<4; i++){
                int[] new_hole = {hole[0]+direct[i][0], hole[1]+direct[i][1]};
                if(new_hole[0] >= border[0]&& new_hole[0] < side && new_hole[1] >= border[1] && new_hole[1] < side){
                    int[][] next_state = new int[side][side];
                    for(int j=0; j<side; j++)
                        for(int k=0; k<side; k++)
                            next_state[j][k] = state[j][k];
                    int save = next_state[hole[0]][hole[1]];
                    next_state[hole[0]][hole[1]] = next_state[new_hole[0]][new_hole[1]];
                    next_state[new_hole[0]][new_hole[1]] = save;
                    String comp = convert(next_state);
                    if(space.containsKey(comp))continue;
                    space.put(comp, i);
                    states.add(next_state);
                    holes.add(new_hole);
                    int work0 = 1, work1 = 1;
                    for(int j=0; j<side; j++){
                        if(next_state[border[0]][j] != side * border[0] + j) work0 = 0;
                        if(next_state[j][border[1]] != side * j + border[1]) work1 = 0;
                    }
                    axis.add(new int[]{border[0]+work0, border[1]+work1});
                }
            }
        }
        ArrayList<Integer> moves = new ArrayList<>();
        int[] cur_space = {side-1, side-1};
        int[][] final_state = new int[side][side];
        for(int i=0; i<side; i++)
            for(int j=0; j<side; j++)
                final_state[i][j] = i * side + j;
        while(space.get(finish)!=-1){
            int cur_move = 3-space.get(finish);
            moves.add(3-cur_move);
            int[]new_space={cur_space[0]+direct[cur_move][0],cur_space[1]+direct[cur_move][1]};
            int save = final_state[cur_space[0]][cur_space[1]];
            final_state[cur_space[0]][cur_space[1]] = final_state[new_space[0]][new_space[1]];
            final_state[new_space[0]][new_space[1]] = save;  
            finish = convert(final_state);
            cur_space = new_space;
        }
        cur_space[0] ++;
        cur_space[1] ++;
        Collections.reverse(moves);
        
        for(int m: moves){
            swap(cur_space[0] + direct[m][0], cur_space[1] + direct[m][1]);
            cur_space[0] = cur_space[0] + direct[m][0];
            cur_space[1] = cur_space[1] + direct[m][1];
        }   
    }
    public void fullSolver(int[][] arr){
        int n = arr.length;
        for(int size = n; size > 3; size--){
            //part1
            ArrayList<Integer> vals_need = new ArrayList<>();
            for(int i=0; i<size; i++) vals_need.add(getId(n-size, n-size+i));
            Collections.reverse(vals_need);
            int cur = n-1;
            for(int i=size/2; i<size; i++){
                bringToLoop(new int[]{n-size, n-size});
                int[] pos = findPosition(getArr(), vals_need.get(i));
                moveOne(pos[0], pos[1], n-size, n-size);

                bringToLoop2(new int[]{cur, n-1}, new int[] {n-size+1, n-size+1});
                moveOne(cur, n-1, n-size, n-size);
                cur--;
            }
            cur = n-2;
            for(int i=size/2-1; i>-1; i--){
                bringToLoop(new int[]{n-size, n-size});
                int[] pos = findPosition(getArr(), vals_need.get(i));
                moveOne(pos[0], pos[1], n-size, n-size);

                bringToLoop2(new int[]{n-1, cur}, new int[] {n-size+1, n-size+1});
                moveOne(n-1, cur, n-size, n-size);
                cur--;
            }
            bringToLoop(new int[]{n-size, n-size});
            finishCycle(n-size, n-size, getArr()[n-(size-size/2)][n-1]);
            //part2
            vals_need.clear();//might break
            for(int i=size-1; i>0; i--) vals_need.add(getId(n-size+i, n-size));
            Collections.reverse(vals_need);
            cur = n-1;
            for(int i=(size-1)/2; i<size-1; i++){
                bringToLoop(new int[]{n-size+1, n-size});
                int[] pos = findPosition(getArr(), vals_need.get(i));
                moveOne(pos[0], pos[1], n-size+1, n-size);

                bringToLoop2(new int[]{cur, n-1}, new int[] {n-size+1+1, n-size+1});
                moveOne(cur, n-1, n-size+1, n-size);
                cur--;
            }
            cur = n-2;
            for(int i=(size-1)/2-1; i>-1; i--){
                bringToLoop(new int[]{n-size+1, n-size});
                int[] pos = findPosition(getArr(), vals_need.get(i));
                moveOne(pos[0], pos[1], n-size+1, n-size);

                bringToLoop2(new int[]{n-1, cur}, new int[] {n-size+1+1, n-size+1});
                moveOne(n-1, cur, n-size+1, n-size);
                cur--;
            }
            bringToLoop(new int[]{n-size+1, n-size});
            finishCycle(n-size+1, n-size, getArr()[n-1][n-((size-1)/2+1)]);
        }
    }
    public void finishCycle(int cornerx, int cornery, int val){
        ArrayList<int[]> thing = new ArrayList<>();
        for(int i=cornerx; i<= side-1; i++) thing.add(new int[]{i, cornery});
        for(int i=cornery+1; i< side-1; i++) thing.add(new int[]{side-1, i});
        for(int i=side-1; i>=cornerx; i--) thing.add(new int[]{i, side-1});
        for(int i=side-1-1; i > cornery; i--) thing.add(new int[]{cornerx, i});

        int[] pos = findPosition(getArr(), val);       
        loop(thing, new int[]{pos[0], pos[1]});
    }
    public void moveOne(int endx, int endy, int cornerx, int cornery){
        //int[] pos = findPosition(getArr(), value);
        int v = getArr()[endx][endy];
        ArrayList<int[]> thing = new ArrayList<>();
        if(endx==cornerx)endx++;
        if(endy==cornery)endy++;
        for(int i=cornerx; i<= endx; i++) thing.add(new int[]{i, cornery});
        for(int i=cornery+1; i< endy; i++) thing.add(new int[]{endx, i});
        for(int i=endx; i>=cornerx; i--) thing.add(new int[]{i, endy});
        for(int i=endy-1; i > cornery; i--) thing.add(new int[]{cornerx, i});
        int[] find = findPosition(getArr(), v);
        loop(thing, find);
    }
    public int[][] getArr(){
        int[][] ret = new int[side][side];
        for(int i=0; i<side; i++)
            for(int j=0; j<side; j++){
                ret[i][j] = matrix[i+1][j+1].toCharArray()[0]-'A';
            }
        return ret;
    }
    public void bringToLoop2(int[] bor, int[] first){
        while(blankR-1!=first[0]){
            if(blankR-1 > first[0]) swap(blankR-1, blankC);
            else swap(blankR+1, blankC);
        }
        while(blankC-1!=first[1]){
            if(blankC-1 > first[1]) swap(blankR, blankC-1);
            else swap(blankR, blankC+1);
        }
        while(blankR-1!=bor[0]){
            if(blankR-1 > bor[0]) swap(blankR-1, blankC);
            else swap(blankR+1, blankC);
        }
        while(blankC-1!=bor[1]){
            if(blankC-1 > bor[1]) swap(blankR, blankC-1);
            else swap(blankR, blankC+1);
        }
        return;
    }
    public void bringToLoop(int[] bor){
        while(blankR-1>bor[0])
            swap(blankR-1, blankC);
        while(blankC-1>bor[1])
            swap(blankR, blankC-1);
        return;
    } //rect is ordered counter clockwise
    public int getId(int r, int c){
        return r * side + c;
    }
    public void loop(ArrayList<int[]> rect, int[] loc){//clockwise movement
        int idx = -1;
        for(int i=0; i<rect.size(); i++)
            if(rect.get(i)[0] == loc[0] && rect.get(i)[1] == loc[1]){
                idx = i;
                break;
            }
        int cur_index = -1;
        for(int i=0; i<rect.size(); i++)
            if(rect.get(i)[0] == blankR-1 && rect.get(i)[1] == blankC-1){
                cur_index = i;
                break;
            }
        //if(opp) idx--;
        cur_index = (cur_index+1)%rect.size();
        for(; idx>0; idx--){
            for(int i=0; i<rect.size()-1; i++){
                swap(rect.get(cur_index)[0]+1, rect.get(cur_index)[1]+1);
                cur_index = (cur_index+1)%rect.size();
            }
        }
        
    }
    public int[] findPosition(int[][] grid, int val){
        for(int i=0; i<grid.length; i++)
            for(int j=0; j<grid[i].length; j++)
                if(grid[i][j] == val)
                    return new int[] {i, j};
        return new int[]{-1, -1};
    }

    public boolean isSolvable(){
        int[][] temp = getArr();
        ArrayList<Integer> flat = new ArrayList<>();
        for(int i=0; i<side; i++)
            for(int j=0; j<side; j++)
                if(temp[i][j] != side*side-1)
                    flat.add(temp[i][j]);
        int cnt = 0;
        for(int i=0; i<flat.size(); i++)
            for(int j=i+1; j<flat.size(); j++)
                if(flat.get(i)>flat.get(j))
                    cnt++;
        if(side%2==1)
            return (cnt%2==0);
        if((side-blankR)%2==0)return (cnt%2==0);
        return (cnt%2==1);
    }
	public String convert(int[][] arr){
        String ret = "";
        for(int i=0; i<arr.length; i++)
            for(int j=0; j<arr[i].length; j++)
                ret += (char)('A' + arr[i][j]);
        return ret;
    }
	
	public boolean okSquare(int r, int c)
	{
		boolean temp = false;
		if (matrix[r-1][c].equals(empty))
			temp = true;
		else if (matrix[r+1][c].equals(empty))
			temp = true;
		else if (matrix[r][c-1].equals(empty))
			temp = true;
		else if (matrix[r][c+1].equals(empty))
			temp = true;
		return temp;	
	}
	
	
	public void swap(int r, int c)
	{
		matrix[blankR][blankC] = matrix[r][c];
		matrix[r][c] = empty;
		blankR = r;
		blankC = c;
		repaint();
        /* 
        try
            {
                Thread.sleep(1);
            }
        catch(InterruptedException ex)
            {
                Thread.currentThread().interrupt();
            }
            */
	}
	
			
	public void update(Graphics g)
	{
		paint(g);
	}
}
