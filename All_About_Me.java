import java.awt.*;
import javax.swing.*;
import javax.swing.Timer;
import java.awt.event.*;
import java.util.*;

public class All_About_Me {
	JFrame frame = new JFrame(); //regular frame for the graphics
	JPanel container = new JPanel(); //will hold all the "slides" within the card layout
	// Buttons to switch between the "slides" (panels) created
	JButton[] slides = new JButton[6];//{new JButton("Home"),new JButton("Origins"),new JButton("Family"),new JButton("Future Life"),new JButton("Dream accomplishment"),JButton("Rule Change")};
    
	//all the "slides" (panels)
	JPanel[] panels = new JPanel[6];//{new JPanel(), new JPanel(), new JPanel(), new JPanel(), new JPanel(), new JPanel()};
    
	JButton addBall = new JButton("Add Ball");
	ArrayList<MovingTextLabel> balls = new ArrayList<>();
	ArrayList<int[]> info = new ArrayList<>();
	public All_About_Me() {
		//container.setLayout(slides); //Set up card layout to have a ppt format in the primary panel which will hold all the rest of the panels (ie the container)
        
        //Add panels for each slide to the primary container of slides
        //Set initial panel
		//slides.show(container, "1"); //this starts the frame by showing the slide labeled '1' within the container in the card layout. 
		
		//frame.add(container); //Officially adds the container of all the frames to the jframe
		//frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); //Makes it so the program ends wehn you close the jframe
        frame.setSize(1800,1000); //sets the size of the frame
		frame.setVisible(true); //visibile
		//texts
        String[] list = {"Home","Origins","Family","Future Life","Influence","Rule Change"};
        String[] sentences = {"<br>Idk because this isn't supposed to be a slide",
                            "<br>Although I have been living in Sugar Land most of my life, I was originally born in Beijing, China.<br> My favorite part of that place was the food, as it featured many delicacies like peking duck and hot pot.",
                            "<br>My family consists of just three people me, my mom, and my dad. Obviously, I was born last.",
                            "<br>Ten years from now, I hope that I would have already graduated from college and began working as an AI coder<br> in games. Hopefully at that time I also established a family.",
                            "<br>My father has had the greatest influence on me.<br> By introducing me to coding as well as inspiring me when times were tough,<br> I have learned to be more resilient and have aspirations in life.",
                            "<br>A change I would make as superintendent would be to have classes teach more life/job skills<br> that would be more applicable to working in the world long term.<br> Also school rankings should only include students who actually go to that school."
							}; 
		Color[] colors = {Color.green, Color.white, Color.CYAN, Color.orange, Color.lightGray, Color.gray};
		for(int i=0; i<6; i++){
			slides[i] = new JButton(list[i]);
			panels[i] = new JPanel();
			slides[i].setVisible(true);
            slides[i].setFont(new Font("Arial", Font.PLAIN, 25));
			panels[i].setBounds(0, 200, 1800, 600);
			panels[i].setBackground(colors[i]);
			panels[i].setVisible(false);
			frame.add(panels[i]);
            JLabel label = new JLabel();
            label.setIcon(new ImageIcon("image"+i+".jpg"));
			label.setVisible(true);
			label.setBounds(200, 350, 1400, 500);
            panels[i].add(label);
            JLabel text = new JLabel();
            text.setText("<html>"+sentences[i]+"<html>");
            text.setFont(new Font("Arial", Font.PLAIN, 35));
            panels[i].add(text);
		}

		for(int i=0; i<6; i++){
			slides[i].setBounds(25+i*300, 50, 250, 80);
			final int idx = i;
            slides[i].addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent e) {
                    for(int j=0; j<6; j++){
                        if(idx != j){
                            panels[j].setVisible(false);
                            slides[j].setVisible(true);
                        }
                        else{
                            panels[j].setVisible(true);
                            slides[j].setVisible(false);
                        }
                    }
				}
			});
			frame.add(slides[i]);
		}
		//make a ball
		addBall.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				MovingTextLabel temp = new MovingTextLabel();
                temp.setVisible(true);
                temp.setBounds(900, 500, 100, 100);
				frame.add(temp);
				balls.add(temp);
				info.add(new int[]{(int)(Math.random()*1800),(int)(Math.random()*1000),(int)(Math.random()*50), (int)(Math.random()*50)});
			}
		});
		for(int i=0; i<balls.size(); i++){
			int[] data = info.get(i);
			//balls.get(i).setBounds(data[0], data[1], 10, 10);
            balls.get(i).setBounds(900, 500, 30, 30);
            
			data[0] += data[2];
			data[1] += data[3];
			if(data[0] < 0 || data[0] > 1800) data[2] *= -1;
			if(data[1] < 0 || data[1] > 1400) data[3] *= -1; 
		}
		addBall.setVisible(true);
		addBall.setBounds(500, 840, 100, 50);
		frame.add(addBall);


		frame.setLayout(null);
	}
	
   //Main Method only needs to call the woooooo method once
    public static void main(String[] args) {
      SwingUtilities.invokeLater(new Runnable() {
		public void run() {
			new All_About_Me();
		}
	  });
   }
}
class MovingTextLabel extends JPanel implements ActionListener {
    private JLabel label;
    public MovingTextLabel() {
        label= new JLabel("NO");
        label.setFont(new Font("Arial", 0, 25));
        label.setBounds(900, 500, 30, 30);
        label.setVisible(true);
        Timer t = new Timer(400, this); // set a timer
        t.start();
        
    }
    public void actionPerformed(ActionEvent e) {
        label.setBounds(900, 500, 300, 300);
        label.setVisible(true);
        label.setText("jasdflojasdlasd");
    }
}

/*
public class All_About_Me {
	JFrame frame = new JFrame(); //regular frame for the graphics
	JPanel container = new JPanel(); //will hold all the "slides" within the card layout
	
	// Buttons to switch between the "slides" (panels) created
	JButton next1 = new JButton("What are Traffic Lights");
	JButton next2 = new JButton("Why you should follow them");
	JButton next3 = new JButton("Importance of Traffic Laws");
    JButton next4 = new JButton("Title");

    
    //all the "slides" (panels)
    JPanel ts = new JPanel();
	JPanel nw = new JPanel();
	JPanel ee = new JPanel();
	JPanel lw = new JPanel();
    
    //creates the card layout
	CardLayout slides = new CardLayout(); //

	public All_About_Me() {
		container.setLayout(slides); //Set up card layout to have a ppt format in the primary panel which will hold all the rest of the panels (ie the container)

        tsPanel a = new tsPanel(); //panel for the title slide (ts)
        frame.addKeyListener(a); //establishes the key listener method within the frame for this specific/ within this specific panel
        frame.addMouseListener(a); //establishes the mouse listener method within the frame for this specific/ within this specific panel
        a.add(next1); //adds the button to swtich between the slides (to be coded later) into this panel

        nwPanel b = new nwPanel(); //panel for the nuclear weapon slide (nw)
		frame.addKeyListener(b); //establishes the key listener method within the frame for this specific/ within this specific panel
		frame.addMouseListener(b); //establishes the mouse listener method within the frame for this specific/ within this specific panel
		b.add(next2); //adds the button to swtich between the slides (to be coded later) into this panel

		eePanel c = new eePanel(); //panel for the enviornmental effects of nucelar bombs (ee)
		frame.addKeyListener(c); //establishes the key listener method within the frame for this specific/ within this specific panel
		frame.addMouseListener(c); //establishes the mouse listener method within the frame for this specific/ within this specific panel
        c.add(next3); //adds the button to swtich between the slides (to be coded later) into this panel

        lwPanel d = new lwPanel(); //panel for the laws of nucelar bombs (lw)
		frame.addKeyListener(d); //establishes the key listener method within the frame for this specific/ within this specific panel
		frame.addMouseListener(d); //establishes the mouse listener method within the frame for this specific/ within this specific panel
        d.add(next4); //adds the button to swtich between the slides (to be coded later) into this panel
        
        //Add panels for each slide to the primary container of slides
		container.add(a, "1");
		container.add(b, "2");
		container.add(c, "3");
		container.add(d, "4");;

        //Set initial panel
		slides.show(container, "1"); //this starts the frame by showing the slide labeled '1' within the container in the card layout. 

        //Button action listeners
        next1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
				CardLayout slides = (CardLayout) container.getLayout();
				a.requestFocusInWindow();
            	slides.next(container);
                }
            });

		next2.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				CardLayout slides = (CardLayout) container.getLayout();
				b.requestFocusInWindow();
            	slides.next(container);
				}
		});

        next3.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				CardLayout slides = (CardLayout) container.getLayout();
				c.requestFocusInWindow();
            	slides.next(container);
				}
		});

		next4.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				CardLayout slides = (CardLayout) container.getLayout();
				d.requestFocusInWindow();
            	slides.next(container);
				}
		});
		
		
		frame.add(container); //Officially adds the container of all the frames to the jframe
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); //Makes it so the program ends wehn you close the jframe
        frame.setSize(2000,1500); //sets the size of the frame
		frame.setVisible(true); //visibile
    }
   //Main Method only needs to call the woooooo method once
   public static void main(String[] args) {
      SwingUtilities.invokeLater(new Runnable() {
		public void run() {
			new All_About_Me();
		}
	  });
   }
}

class tsPanel extends JPanel implements ActionListener, MouseListener, KeyListener
{
	private int numKey;
	private Timer time;
	private int change;
	private int x,y;
	private int add;
	private int rx,ry;
	private int lastx, lasty, speedx, speedy, dec;
	private long lastTime;
	private boolean isOnSquare;
	private int lightPos;

	public tsPanel()
	{

		setSize(2000, 1500);
		x=0;
		y=620;
		add=5;
		lastx = lasty = speedx = speedy = 0;
		lastTime = System.currentTimeMillis();
		dec = 1;
		rx=ry=300;
		time = new Timer(15, this); //sets delay to 15 milliseconds and calls the actionPerformed of this class.
		time.start();
		change=1;
		setVisible(true);
		addKeyListener(this);
		addMouseListener(this);
		setFocusable(true); //very important since it forces your key listeners to work here
	}


	public void paintComponent(Graphics g)
	{
		switch (change)
		{
			case 1: 	g.setColor(Color.WHITE); break;
			case 2: 	g.setColor(Color.WHITE); break;
			case 3: 	g.setColor(Color.WHITE); break;
		}
		background(g); // sets up background
		Color myColor1 = new Color(229, 76, 56);
		myCar(g,x,y,myColor1);
		myColor1 = new Color(167, 199, 231);
		myCar(g,x+300,y+100,myColor1); // draws car
		myLight(g,lightPos);
		myRect(g,rx,ry);
		drawText(g);
	}

	public void actionPerformed(ActionEvent e)
	{
		if (x >=getWidth())
			x=1;
		if(change ==1){
			x+=add;
		}
		if (isOnSquare){
		rx += 1*speedx;
		ry += 1*speedy;
		}
		if(rx<=0 || rx >= getWidth()){
			speedx*=-1;
		}
		if(ry >= getHeight() || ry <= 0){
			speedy*=-1;
		}		
	
		repaint();
	}
	public void background(Graphics g) {
		//sets screen to a white background
		g.setColor(Color.WHITE);
		g.fillRect(0,0,getWidth(),getHeight());
		
		//road
		g.setColor(Color.BLACK);
		g.fillRect(0,600,getWidth(),200);

		
		//sky 
		Color myColor = new Color(143, 177, 204);
		g.setColor(myColor);
		g.fillRect(0,0,getWidth(),600);
		
		//Grass
		Color myColor3 = new Color(119, 221, 118);
		g.setColor(myColor3);
		g.fillRect(0,800,getWidth(),getWidth()-800);
		
		//White Lines on road
		g.setColor(Color.WHITE);
		for (int i = 50; i<getWidth();i+=150){
				g.fillRect(i,690,100,20);
		}
	}
	
	public void drawText(Graphics g) {
		g.setColor(Color.BLACK); //title text
		g.setFont(new Font ("SANS_SERIF", Font.BOLD, 60));
		g.drawString("TRAFFIC LIGHTS", 150, 100);
		g.setFont(new Font ("SANS_SERIF", Font.BOLD, 15));
		g.drawString("Project by Shayaan Sameer and Ziduo Yi", 150, 125);	
		g.drawString("Click 1 to have the light change to red and 2 for it to change back to greeen!", 150, 150);
		g.drawString("Use your arrow keys to move the Traffic light around in the left or right direction!", 150, 170);
		g.drawString("Also you can throw the sun around", 150, 190);
				
	}
	public void myRect(Graphics g,int x, int y)
	{		
		g.setColor(Color.yellow);
		g.fillOval(x, y, 160, 160);
	}
	public void myLight(Graphics g, int move)
	{
		g.setColor(Color.GRAY);
		g.fillRect(1200+lightPos,80,160,300);
		g.setColor(Color.BLACK);
		if (change == 1){
			g.setColor(Color.GREEN);
		}
		g.fillOval(1240+lightPos,85,80,80);//toplight
		g.setColor(Color.BLACK);
		g.fillOval(1240+lightPos,185,80,80);//middle light
		if (change == 2){
			g.setColor(Color.RED);
		}
		g.fillOval(1240+lightPos,285,80,80);//bottom light
	}		
	 
	public void myCar(Graphics g, int x, int y, Color myColor1){ 
	
		
		g.setColor(myColor1);
		g.fillRect(x,y,250, 40);
		
		int[]x1 = {x+25,x+50,x+150,x+190};
		int[]y1 = {y,y-25,y-25,y};
		       
		g.setColor(myColor1);
		g.fillPolygon(x1,y1,4);
		
		g.setColor(Color.GRAY);
		g.fillOval(x+30,y+13, 40, 40);
		g.setColor(Color.DARK_GRAY);
		g.fillOval(x+35,y+18, 30, 30);
		
		g.setColor(Color.GRAY);
		g.fillOval(x+180,y+13, 40, 40);
		g.setColor(Color.DARK_GRAY);
		g.fillOval(x+185,y+18, 30, 30);
		
		int[]x2 = {x+50,x+65,x+140,x+165};
		int[]y2 = {y,y-15,y-15,y};
		
		g.setColor(Color.LIGHT_GRAY);
		g.fillPolygon(x2,y2,4);
		
		g.setColor(myColor1);
		g.fillRect(x+105,y-15,5,15);
		
		x=x+50;


	}

	public void mousePressed(MouseEvent e){
		lastx = e.getX();
		lasty = e.getY();
		if (lastx>=rx && lastx <=rx-20+115 && lasty>=ry && lasty<=ry-20+115){
			isOnSquare= true;
		} else isOnSquare = false;
		lastTime = System.currentTimeMillis();
		
	}

	public void mouseReleased(MouseEvent e){
		int timeTaken = (int)(System.currentTimeMillis()-lastTime);
		speedx = (e.getX()-lastx)/timeTaken * 5;
		speedy = (e.getY()-lasty)/timeTaken * 5;
	}
	public void mouseEntered(MouseEvent e){

	}
	public void mouseExited(MouseEvent e){
	
	}

	public void keyPressed(KeyEvent e)
	{
		  int code = e.getKeyCode();
		  
		  if(code == KeyEvent.VK_RIGHT) { // moves car right if right key pressed
			  lightPos+=20;
		  }
		  if(code == KeyEvent.VK_LEFT) { // moves car left if left key pressed
			  lightPos-=20;
		  }

		  switch (code)
		  {
		  	case '1': change = 1; break;
		  	case '2': change = 2; break;
		  }

       *code == KeyEvent.VK_RIGHT for right arrow
        * code == KeyEvent.VK_LEFT for left arrow
        * code == KeyEvent.VK_SPACE
        *

	}
	public void mouseClicked(MouseEvent e)
	{
	}

	public void keyReleased(KeyEvent e)	{}
	public void keyTyped(KeyEvent e){}


}

class nwPanel extends JPanel implements ActionListener, MouseListener, KeyListener
{
	private Timer time;
	private int change = 1;
	private int numKey;
	private int x;
	private int add;
	private boolean isOnRed;
	private boolean isOnGreen;
	private boolean isOnYellow;
	public nwPanel()
	{	
		x=0;
		add=5;
		time = new Timer(15, this); //sets delay to 15 millis and calls the actionPerformed of this class.
		setSize(2000, 1500);
		setVisible(true); //it's like calling the repaint method.
		time.start();
		change=1;
		addKeyListener(this);
		addMouseListener(this);
		setFocusable(true);
	}

    public void paintComponent(Graphics g) {
		switch (change) // change background colors
		{
			case 1: 	g.setColor(new Color(143, 177, 204)); break;
			case 2:  	g.setColor(new Color(143, 177, 204)); break;
			case 3:  	g.setColor(new Color(143, 177, 204)); break;
		}
		// background
		g.fillRect(0, 0, 2000, 1500);
		climatePanelText(g);
		if(numKey>=3){
			myPerson(g);
			myLights(g);
		}
		
		
    }

	public void actionPerformed(ActionEvent e)
	{
		if (x >=getWidth())
			x=1;
		x+=add;
		if(isOnRed){
			x-=add;
		}
		if(isOnYellow){
			x-=4;
		}
		
		repaint();
	}
	public void climatePanelText(Graphics g) {
		// text - title
		g.setColor(Color.BLACK);
		g.setFont(new Font ("Arial", Font.BOLD, 40));
		g.drawString("WHAT ARE TRAFFIC LIGHTS", 700,65);
				
		// first paragraph
		g.setFont(new Font ("Arial", Font.PLAIN, 25));
		g.setColor(new Color(0,153,0));
				
		if(numKey == 0) {
			g.drawString("Click!", 50, 100);
		}
		if(numKey >= 1) {
			g.setColor(Color.BLACK);
			g.setFont(new Font ("Arial", Font.PLAIN, 25));
			g.drawString("Traffic lights are a common sight on roads and intersections, serving as important devices for regulating vehicular and pedestrian traffic.", 50,100);
			g.drawString("They consist of multiple colored lights housed in a vertical or horizontal arrangement. Typically, a traffic light includes three lights: red, yellow, and green.", 50,140);
			g.drawString("Each light represents a specific instruction for drivers and pedestrians. The red light signals vehicles and pedestrians to stop and wait. ", 50,180);
		}				
		if(numKey >= 2) {
			g.setColor(Color.BLACK);
			g.setFont(new Font ("Arial", Font.PLAIN, 25));
			g.drawString("The yellow light indicates that the light is about to change, signaling caution to both drivers and pedestrians.", 50,220);
			g.drawString("Finally, the green light grants permission for vehicles to proceed, while pedestrians are instructed to cross the road.", 50,260);
			g.drawString("Traffic lights play a crucial role in maintaining order and efficiency in urban traffic systems, reducing accidents and improving overall road safety.", 50,300);
		}
		if(numKey >= 3) {
			g.setColor(Color.CYAN);
			g.drawString("To make sure you understand which color means, click on the color of the light and see how the person reacts!", 50,355); 
		}
		

	}
	public void myLights(Graphics g) {
		g.setColor(Color.GREEN);
		g.fillOval(700,900,100,100);
		g.setColor(Color.YELLOW);
		g.fillOval(900,900,100,100);
		g.setColor(Color.RED);
		g.fillOval(1100,900,100,100);
 
	}
	public void myPerson(Graphics g) {
        // Draw person
        g.setColor(Color.BLACK);

        // Head
        g.drawOval(50+x, 400, 100, 100);

        // Body
        g.drawLine(100+x, 500, 100+x, 700);

        // Arms
        g.drawLine(30+x, 550, 170+x, 550);

        // Legs
        g.drawLine(100+x, 700, 70+x, 850);
        g.drawLine(100+x, 700, 130+x, 850);

        // Face
        g.drawArc(70+x, 425, 30, 30, 180, 180); // Left eye
        g.drawArc(100+x, 425, 30, 30, 180, 180); // Right eye
        g.drawArc(75+x, 450, 50, 40, 180, 180); // Mouth

        // Hair
        g.setColor(new Color(139, 69, 19)); // Brown color
        g.fillArc(50+x, 395, 100, 40, 0, 180);

        // Shirt
        g.setColor(Color.BLUE);
        g.fillRect(80+x, 500, 40, 200);

        // Pants
        g.setColor(Color.GRAY);
        g.fillRect(80+x, 700, 40, 100);

        // Shoes
        g.setColor(Color.BLACK);
        g.fillRect(70+x, 850, 30, 10);
        g.fillRect(100+x, 850, 30, 10);
	}


	public void mouseClicked(MouseEvent e)
	{
		numKey++;
		if (e.getX()>=1000 && e.getX()<=1200 && e.getY()>=800 && e.getY()<=1000){
			isOnRed = true;
		}else isOnRed = false;
		
		if (e.getX()>=600 && e.getX()<=800 && e.getY()>=800 && e.getY()<=1000){
			isOnGreen = true;
		}else isOnGreen = false;
		
		if (e.getX()>=800 && e.getX()<=1000 && e.getY()>=800 && e.getY()<=1000){
			isOnYellow = true;
		}else isOnYellow = false;
	}
	public void mousePressed(MouseEvent e){}
	
	public void mouseReleased(MouseEvent e){

	}

	public void mouseEntered(MouseEvent e){}
	public void mouseExited(MouseEvent e){}


	@Override 
	public void keyPressed(KeyEvent e) {
        int keyCode = e.getKeyCode();
		switch (keyCode)
		  {
			case KeyEvent.VK_1: change = 1; break;
			case KeyEvent.VK_2: change = 2; break;
			case KeyEvent.VK_3: change = 3; break;
		  }
    }
	
	public void keyReleased(KeyEvent e) {

	}
    @Override public void keyTyped(KeyEvent e) {}
}

class eePanel extends JPanel implements KeyListener, ActionListener, MouseListener {

	private Timer time;
	private int add;
	private int change;
	private boolean isOn;
	private int x,y;
	private boolean crash;
	private int count;


	public eePanel()
	{
		time = new Timer(15, this); //sets delay to 15 millis and calls the actionPerformed of this class.
		setSize(2000, 1500);
		setVisible(true); //it's like calling the repaint method.
		time.start();
		add = 10;
		change=1;
		x=0;
		y=760;
		count=0;
		addKeyListener(this);
		addMouseListener(this);
		setFocusable(true);

	}

	public void paintComponent(Graphics g)
	{
		switch (change)
		{
			case 1: 	g.setColor(new Color(143, 177, 204)); break;
			case 2: 	g.setColor(new Color(143, 177, 204)); break;
			case 3: 	g.setColor(new Color(150, 0,150)); break;
		}

		
		g.fillRect(0, 0, 2000, 1500);
		g.setColor(Color.BLACK);
		
		g.setFont(new Font ("Arial", Font.PLAIN, 25));
		g.drawString("Traffic lights serve as a vital tool for maintaining order and ensuring the smooth flow of traffic. They provide a standardized system that regulates the movement", 20,100);
		g.drawString("of vehicles and pedestrians, minimizing the risk of accidents and promoting safety on the roads. By adhering to traffic lights, drivers can anticipate and respond ", 20,140);
		g.drawString("to changing traffic conditions, reducing the likelihood of collisions and conflicts at intersections. Obeying traffic lights promotes fairness and equality among road users. ", 20,180);
		g.drawString("Traffic lights allocate right-of-way based on a predetermined sequence, treating all vehicles and pedestrians equally. This impartiality ensures that everyone gets", 20,220);
		g.drawString("a fair chance to proceed and minimizes disputes or conflicts that may arise from individual judgments. By following traffic lights you create a safer enviornment", 20,260);
		g.drawString("for everyone", 20,300);
		
		g.drawString("Click the right arrow key to see an example of what can happen when you dont follow traffic lights", 20,400);
		

		if (change==2){
			myLight(g);
			myCar(g,x,y);
			if(crash){
				myExplosion(g);
				g.drawString("YOU CRASH!!", 100,800);
			}
		}

        
	}
	public void myExplosion(Graphics g) {
		// Set explosion parameters
        int centerX = getWidth() / 2;
        int centerY = getHeight() / 2;
        int numParticles = 50;
        int particleSize = 10;

        // Draw explosion particles
        g.setColor(Color.RED);
        for (int i = 0; i < numParticles; i++) {
            int x = (int) (Math.random() * getWidth());
            int y = (int) (Math.random() * getHeight());
            g.fillOval(x, y, particleSize, particleSize);
        }
 
	}
	public void myLight(Graphics g)
	{
		g.setColor(Color.GRAY);
		g.fillRect(1200,600,80,150);
		g.setColor(Color.BLACK);
		g.fillOval(1220,605,40,40);//toplight
		g.fillOval(1220,655,40,40);//middle light
		g.setColor(Color.RED);
		g.fillOval(1220,705,40,40);//bottom light
	}		
	 
	public void myCar(Graphics g, int x, int y){ 
	
		Color myColor1 = new Color(229, 76, 56);
		g.setColor(myColor1);
		g.fillRect(x,y,250, 40);
		
		int[]x1 = {x+25,x+50,x+150,x+190};
		int[]y1 = {y,y-25,y-25,y};
		       
		g.setColor(myColor1);
		g.fillPolygon(x1,y1,4);
		
		g.setColor(Color.GRAY);
		g.fillOval(x+30,y+13, 40, 40);
		g.setColor(Color.DARK_GRAY);
		g.fillOval(x+35,y+18, 30, 30);
		
		g.setColor(Color.GRAY);
		g.fillOval(x+180,y+13, 40, 40);
		g.setColor(Color.DARK_GRAY);
		g.fillOval(x+185,y+18, 30, 30);
		
		int[]x2 = {x+50,x+65,x+140,x+165};
		int[]y2 = {y,y-15,y-15,y};
		
		g.setColor(Color.LIGHT_GRAY);
		g.fillPolygon(x2,y2,4);
		
		g.setColor(myColor1);
		g.fillRect(x+105,y-15,5,15);
		
		x=x+50;


	}

	public void actionPerformed(ActionEvent e)
	{
		if (x >=getWidth())
			crash = true;
		if(change == 2){
			x+=add;
		}
		repaint();
	}


	public void mouseClicked(MouseEvent e)
	{

	}
	public void mousePressed(MouseEvent e){

	}
	
	public void mouseReleased(MouseEvent e){}
	public void mouseEntered(MouseEvent e){}
	public void mouseExited(MouseEvent e){}


	public void keyPressed(KeyEvent e)
	{
		int code = e.getKeyCode();


		  switch (code)
		  {
			case KeyEvent.VK_1: change = 1; break;
			case KeyEvent.VK_RIGHT: change = 2; break;
			case KeyEvent.VK_3: change = 3; break;
		  }

       *code == KeyEvent.VK_RIGHT for right arrow
        * code == KeyEvent.VK_LEFT for left arrow
        * code == KeyEvent.VK_SPACE
        *

	}

	public void keyReleased(KeyEvent e)	{}
	public void keyTyped(KeyEvent e){}
	
}

class lwPanel extends JPanel implements KeyListener, ActionListener, MouseListener
{
	private Timer time;
	private int add;
	private int change;

	public lwPanel()
	{

		time = new Timer(15, this); //sets delay to 15 millis and calls the actionPerformed of this class.
		setSize(2000, 1500);
		setVisible(true); //it's like calling the repaint method.
		time.start();
		add = 1;
		change=1;
		addKeyListener(this);
		addMouseListener(this);
		setFocusable(true);
	}


	public void paintComponent(Graphics g)
	{
		switch (change)
		{
			case 1: 	g.setColor(Color.LIGHT_GRAY); break;
			case 2: 	g.setColor(new Color(150, 150, 0)); break;
			case 3: 	g.setColor(new Color(150, 0,150)); break;
		}

		g.fillRect(0, 0, 2000, 1500);
		Font font = new Font("Arial", Font.PLAIN, 25);
        g.setFont(font);

        // Set text color
        g.setColor(Color.BLACK);

        // Traffic Laws and Traffic Lights Paragraph
        g.drawString("Traffic laws and traffic lights play a crucial role in ensuring the safety and orderliness of our roads. These regulations are designed to govern the behavior of drivers,",20,100);
        g.drawString(" pedestrians, and other road users. They provide a standardized framework that promotes uniformity and predictability in traffic movements. By obeying traffic laws,",20,140);
        g.drawString( " drivers contribute to the overall safety of themselves and others. Traffic lights, in particular, serve as essential control devices at intersections. ",20,180);
        g.drawString("They allocate the right-of-way, signaling when drivers should stop, yield, or proceed with caution. By following the signals displayed by traffic lights,",20,220);
        g.drawString(" drivers can navigate intersections smoothly, minimizing the risk of accidents and conflicts. Pedestrians also rely on traffic lights to safely cross the road. ",20,260);
        g.drawString( "Crosswalk signals indicate when it's safe to walk, ensuring the protection of pedestrians. Moreover, traffic laws and traffic lights promote fairness and equality among",20,300);
        g.drawString( "road users. By adhering to these regulations, everyone has an equal chance to use the road, regardless of their mode of transportation. This promotes a sense of ",20,340);
        g.drawString( "respect, cooperation, and shared responsibility. Traffic laws and traffic lights also contribute to the efficient flow of traffic. They help manage congestion by ",20,380);
        g.drawString( "organizing the movement of vehicles, minimizing disruptions and delays. Without these regulations, chaos would ensue, leading to increased accidents, gridlock,",20,420);
        g.drawString(" and frustration. In conclusion, traffic laws and traffic lights are essential for maintaining safe and orderly roads. They protect lives, promote fairness, ",20,460);
        g.drawString(" and contribute to the overall efficiency of our transportation system.",20,500);


		
        
	}

	public void actionPerformed(ActionEvent e)
	{
		repaint();
	}


	public void mouseClicked(MouseEvent e)
	{
	}
	public void mousePressed(MouseEvent e){

	}
	
	public void mouseReleased(MouseEvent e){}
	public void mouseEntered(MouseEvent e){}
	public void mouseExited(MouseEvent e){}


	public void keyPressed(KeyEvent e)
	{
		int code = e.getKeyCode();

		  switch (code)
		  {
			case KeyEvent.VK_1: change = 1; break;
			case KeyEvent.VK_2: change = 2; break;
			case KeyEvent.VK_3: change = 3; break;
		  }

       *code == KeyEvent.VK_RIGHT for right arrow
        * code == KeyEvent.VK_LEFT for left arrow
        * code == KeyEvent.VK_SPACE
        *

	}

	public void keyReleased(KeyEvent e)	{}
	public void keyTyped(KeyEvent e){}
	
}

*/