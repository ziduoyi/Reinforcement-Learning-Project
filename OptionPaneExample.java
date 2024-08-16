import javax.swing.*;  
public class OptionPaneExample {  
    JFrame f;  
    int width;
    OptionPaneExample(){  
        f=new JFrame();   
        while (1 > 0){
            String number=JOptionPane.showInputDialog(f,"2 - 2x2\n3 - 3x3\netc.\n(min is a 2x2, max is a 8x8)\nEnter Grid Size:");    
            if(number.matches("\\d+")){
                int num = Integer.parseInt(number);
                width = num;
                if (num > 1 && num < 101)
                    break;
            }
        }
    }  
}  