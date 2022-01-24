///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
//     ==== There is no need to change this file ====
// 

#include "ViewAgent.h"
#include "MainWindow.h"
#include <QGraphicsItemAnimation>

#include <stdlib.h>

/* XPM */static const char *bgt[] = {"33 18 250 2 ", "   c None", ".  c #010101", "X  c #030303", "o  c #030404", "O  c #030405", "+  c #020406", "@  c #040404", "#  c #040405", "$  c #050505", "%  c #070504", "&  c #050607", "*  c #060606", "=  c #070606", "-  c #060707", ";  c #070707", ":  c #080605", ">  c #0A0806", ",  c #0B0806", "<  c #0A0807", "1  c #0C0806", "2  c #0C0906", "3  c #0D0906", "4  c #0C0907", "5  c #030508", "6  c #050708", "7  c #070809", "8  c #06080A", "9  c #080808", "0  c #0B0C0E", "q  c #0C0C0D", "w  c #0D0E0F", "e  c #120D04", "r  c #120D06", "t  c #160E08", "y  c #171006", "u  c #1B120A", "i  c #0C0E10", "p  c #0F1113", "a  c #111112", "s  c #121213", "d  c #101114", "f  c #151515", "g  c #151616", "h  c #18191A", "j  c #1A1B1D", "k  c #24170D", "l  c #25180D", "z  c #26180D", "x  c #26180E", "c  c #311F10", "v  c #392511", "b  c #3F2815", "n  c #212222", "m  c #272829", "M  c #29292A", "N  c #39393B", "B  c #492F19", "V  c #4E321A", "C  c #4F321A", "Z  c #50331A", "A  c #51331B", "S  c #51341B", "D  c #52341B", "F  c #56361C", "G  c #6B4813", "H  c #63461A", "J  c #764F16", "K  c #684222", "L  c #6B4423", "P  c #7D5025", "I  c #444446", "U  c #454647", "Y  c #515151", "T  c #515152", "R  c #535556", "E  c #565759", "W  c #796C5A", "Q  c #646466", "!  c #707071", "~  c #7B7A79", "^  c #B3791E", "/  c #80512A", "(  c #81522B", ")  c #82532B", "_  c #84542A", "`  c #84542B", "'  c #86552A", "]  c #88572A", "[  c #89582A", "{  c #8F5B2E", "}  c #936026", "|  c #986032", " . c #A76A37", ".. c #A96C36", "X. c #AC6F35", "o. c #AD6F35", "O. c #AF7134", "+. c #B27433", "@. c #B37433", "#. c #B47337", "$. c #BA7739", "%. c #B9773B", "&. c #BC793A", "*. c #C17D38", "=. c #9B7D51", "-. c #B6802C", ";. c #D18D2A", ":. c #D68F2E", ">. c #D7902D", ",. c #DB942B", "<. c #DF9729", "1. c #DD962A", "2. c #DF972A", "3. c #DA922C", "4. c #DB932D", "5. c #D8912E", "6. c #DB942C", "7. c #DD952C", "8. c #DC942D", "9. c #C68135", "0. c #C58136", "q. c #C98434", "w. c #CA8534", "e. c #CB8534", "r. c #CA8435", "t. c #CE8832", "y. c #C48038", "u. c #D08A31", "i. c #D08A32", "p. c #D18A32", "a. c #D38C31", "s. c #D48D30", "d. c #D58E30", "f. c #E09623", "g. c #E09624", "h. c #E19724", "j. c #E09725", "k. c #E19725", "l. c #E19825", "z. c #E09826", "x. c #E19826", "c. c #E19827", "v. c #E19927", "b. c #E09828", "n. c #E19928", "m. c #E09829", "M. c #E29928", "N. c #E19A28", "B. c #E19A29", "V. c #E29A28", "C. c #E29A29", "Z. c #E29B29", "A. c #E2992A", "S. c #E19A2A", "D. c #E19B2B", "F. c #E49B28", "G. c #E49B29", "H. c #E29C2B", "J. c #E59C28", "K. c #E29C2C", "L. c #E39C2D", "P. c #E29C2E", "I. c #E29E2F", "U. c #E39E2F", "Y. c #E59D2C", "T. c #E29E30", "R. c #E59F31", "E. c #E2A133", "W. c #E3A539", "Q. c #E4A538", "!. c #E4A63A", "~. c #E4A73C", "^. c #E5A43E", "/. c #E7A53E", "(. c #85807A", "). c #84817C", "_. c #AF956D", "`. c #B49F7F", "'. c #E5AB41", "]. c #E7A944", "[. c #E6AE46", "{. c #E8AC40", "}. c #E8AD41", "|. c #E9B046", " X c #E7B24B", ".X c #E7B34C", "XX c #E7B44D", "oX c #EAB148", "OX c #EAB14B", "+X c #E9B44D", "@X c #EBB64C", "#X c #ECB74F", "$X c #E9AE52", "%X c #E8B752", "&X c #EAB154", "*X c #E8BA55", "=X c #E9BC59", "-X c #EEBF59", ";X c #ECB561", ":X c #EBB664", ">X c #EFBF6A", ",X c #EBBB71", "<X c #EAC05D", "1X c #EFC35F", "2X c #F0C35D", "3X c #F0C45F", "4X c #ECC261", "5X c #EECB6D", "6X c #EFCD6F", "7X c #F1C763", "8X c #F3CD6B", "9X c #EFC27C", "0X c #EFC27D", "qX c #F2D274", "wX c #F3D97D", "eX c #F4DA7E", "rX c #878888", "tX c #959492", "yX c #AEA290", "uX c #AEA598", "iX c #AAA9A8", "pX c #B1ACA1", "aX c #B7B6B2", "sX c #B8B6B1", "dX c #BBB6B0", "fX c #C2BFB5", "gX c #F2C986", "hX c #F7DE82", "jX c #F2CE96", "kX c #F5DD98", "lX c #F7E085", "zX c #F8E387", "xX c #FAE588", "cX c #FAE68B", "vX c #FAE78B", "bX c #FCE890", "nX c #DDD5B8", "mX c #DAD0BD", "MX c #E9CEA7", "NX c #EBD4B0", "BX c #EED7B1", "VX c #EAD7B9", "CX c #FFEFA4", "ZX c #FFF0AD", "AX c #F7E9BA", "SX c #F8ECB8", "DX c #D3CDC2", "FX c #E6D9C6", "GX c #F6EBC6", "HX c #F5EED5", /* pixels */ "                                                . o 0 p d i 5     ", "                                          . w m Y (.yX`._.=.H y   ", "                                    X g U ~ pXnXSX>X/.R.Y.Y.G.J = ", "                                ; M ! fXNX0X&X4XxX@Xj.n.n.n.A.} : ", "                          . a I tXmXAXCXoXf.h.W.eX7Xc.n.n.M.a._ > ", "                      $ n Q sXVX9X].<XvX2Xc.n.I.6X-Xc.n.M.:.$.S * ", "                X q N rXDXGXZXOXg.k.'.lX3Xc.n.D.%XQ.v.<.p.o.F 3   ", "            X s T iXFXjX$X=XzX2Xc.v.E.qX|.x.n.N.H.n.3.0.{ c @     ", "        @ f E aXHXgX^.x.h.~.wX8Xn.n.K..XU.v.V.<.>.e.#.K t O       ", "      - R dXBXkXbX}.j.n.v.T.5X1Xv.n.n.C.n.,.u.y.%.| b %           ", "    h ).MX;XP.*XcX#Xz.n.n.S.XX!.v.M.<.:.r.&. ./ V u O             ", "  j uX,XL.l.x. XhX{.c.n.n.n.Z.b.6.i.*...( C k , 7                 ", "9 W :Xx.v.n.v.[.+XB.n.V.n.1.d.9.X.) Z l , 6                       ", "r -.F.n.n.n.n.K.N.M.2.5.w.O.` A l 1 6                             ", "e ^ J.M.M.V.n.m.4.t.+.' S z 2 6                                   ", "= G ;.7.8.s.q.@.] D x 2 &                                         ", "  4 v P [ L B x 1 &                                               ", "    8 < , # +                                                     " };


ViewAgent::ViewAgent(Ped::Tagent * agent, QGraphicsScene * scene) : agent(agent)
{
	QBrush greenBrush(Qt::green);
	QPen outlinePen(Qt::black);
	outlinePen.setWidth(2);


#ifdef TEACHER_IS_FRENCH 
/* What do you call a small bug? */
	
	bgt_icon = scene->addPixmap(QPixmap(bgt).scaled(20, 20));
	bgt_icon->setPos(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()));
#else
	bgt_icon = 0;
	rect = scene->addRect(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()),
	MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1, outlinePen, greenBrush);
#endif
}

void ViewAgent::paint(QColor color){
	
	if(bgt_icon)
		bgt_icon->setPos(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()));
	else
	{
		QBrush brush(color);
		rect->setBrush(brush);
		rect->setRect(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()),
		MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1);

	}

	
}

const std::pair<int, int> ViewAgent::getPosition(){
	return std::make_pair(agent->getX(), agent->getY());
}


