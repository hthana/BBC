
#include "stdafx.h"
#include "Timer.h" 
#include "WormTrack.h" 

using namespace std;
using namespace cv;

int RefineCurvature(seg* segment, vector<Point>& cur);
int DrawImage(seg& segment, Mat& I, int partnum, vector<Point> cur);

int CurvatureAnalysis(vector<double>* curvature, vector<Point>& cur);
int RefineCurvature(seg* segment, vector<Point>& cur);
int FindMax(vector<double>* input, int start, int end);
int FindMin(vector<double>* input, int start, int end);
int FindPerpPoint(vector<Point>* input, Point x, Point targent, Point &result, int startindex, int endindex);
void Smooth1DSequence(const vector<double>* input, vector<double>& output, double sigma);


Point m_A = Point(0, 0);

Point m_B = Point(480, 360);
Point head = Point(0, 0);
Point tail = Point(0, 0);

int main()
{
	myTimer Timer; 
	seg segment; 
	

	segment.head = head;
	segment.tail = tail;

	ofstream file("Pharynx.csv");
	ofstream file1("PeakPoints.csv");
	ofstream file2("InflectionPoints.csv");
	for (int j = 1; j <= 200; j++) 
	{

		string folder = "samples\\";
		string n, b;
		n = to_string(j);
		b = folder + n + ".png";
		string name(b);
	    Mat I = imread(name, 0); 
		Mat I_color = imread(name, 1); 


		int guassian = 1;
		int threshold = 70;
		int morpology = 1;
		bool switchht = 0;
		int partnum = 120;


		WormTrack Track(I, 1, guassian, threshold, morpology, partnum, m_A, m_B, &segment, switchht);
		
		string s = "time cost for image " + to_string(j);
		Timer.StartTimer(s.c_str()); 


		Track.PreProcess();
		Track.Contour();
		Track.Analysis();
		Track.Curvaturecalculation();
		Track.Lengthcalculation();
		Track.Widthcalculation();
		Track.Volumecalculation();
		Track.Segment();

		Timer.StopTimer();
		
		segment = Track.segment; 

	
		vector<Point> cur;
		RefineCurvature(&segment, cur); 
		
		DrawImage(segment, I_color, partnum, cur);
		
		folder += "analysis results\\";
		string a;
		a = folder + n + "_.png";
		string c(a);
		imwrite(c, I_color);
		head = segment.head;
		tail = segment.tail;
		m_A = Track.segment.topleft - Point(50, 50);
		m_B = Track.segment.botright + Point(50, 50);
		cout << "segment.center: " << segment.center[12] << "\n";
		
		
		if (file)
		{
			file << segment.center[12].x << "," << segment.center[12].y;
		}
	
		for (int i = 0; i < cur.size(); i++)
		{
			if (cur[i].x == 0)
			{

				cout << "inflection point: " << segment.center[cur[i].y] << "\n";
				if (file2)
				{
					file2 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
			else if (cur[i].x == -1)
			{
			
				cout << "peak point: " << segment.center[cur[i].y] << "\n";
				if (file1)
				{
					file1 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
			else
			{

				cout << "peak point: " << segment.center[cur[i].y] << "\n";
				if (file1)
				{
					file1 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
		}


		if (file)
		{
			file <<  "\n";
		}
		
		if (file1)
		{
			file1 << "\n";
		}

		if (file2)
		{
			file2 << "\n";
		}


		Track.~WormTrack();
	}
	file.close();
	file1.close();
	file2.close();
	waitKey();
	return 0;
}


int RefineCurvature(seg* segment, vector<Point>& result)
{
	if (segment->curvatureC.size() == 0)
	{
		return 0;
	}

	vector<Point> curC;
	vector<double> curvatureA, curvatureB, scurvA, scurvB;
	CurvatureAnalysis(&(segment->curvatureC), curC);

	if (curC.size() < 5)
	{
		return 0;
	}

	curvatureA.insert(curvatureA.begin(), (segment->curvatureA).begin() + 1, (segment->curvatureA).end() - 1);
	curvatureB.insert(curvatureB.begin(), (segment->curvatureB).begin() + 1, (segment->curvatureB).end() - 1);

	Smooth1DSequence(&(curvatureA), scurvA, 3); 
	Smooth1DSequence(&(curvatureB), scurvB, 3); 


	int a, b; 
	a = 0;

	int length = ((segment->Partnum / curC.size()) > 1 ? (segment->Partnum / curC.size()) : 1) * 1.5;

	Point backward, forward, tangent;
	int step = 10; 
	for (int i = 0; i < curC.size(); i++)
	{
		b = (a + length) < (segment->center.size() - 1) ? (a + length) : (segment->center.size() - 1);

		if (curC[i].x == 0)
		{
			result.push_back(curC[i]);
		}
		else if (curC[i].x == 1)
		{
			int start = (curC[i].y - length) > 0 ? (curC[i].y - length) : 0;
			int end = (curC[i].y + length) < scurvA.size() ? (curC[i].y + length) : scurvA.size();

			Point perp;
			int index = segment->tabA[1 + FindMax(&scurvA, start, end)].y;

			backward = segment->contourA[(index - step) > 0 ? (index - step) : 0];
			forward = segment->contourA[(index + step) < (segment->contourA.size() - 1) ? (index + step) : (segment->contourA.size() - 1)];
			tangent = forward - backward;

			result.push_back(Point(1, FindPerpPoint(&(segment->center), segment->contourA[index], tangent, perp, a, b)));
		}
		else
		{
			int start = (curC[i].y - length) > 0 ? (curC[i].y - length) : 0;
			int end = (curC[i].y + length) < scurvB.size() ? (curC[i].y + length) : scurvB.size();

			Point perp;
			int index = segment->tabB[1 + FindMin(&scurvB, start, end)].y;

			backward = segment->contourB[(index - step) > 0 ? (index - step) : 0];
			forward = segment->contourB[(index + step) < (segment->contourB.size() - 1) ? (index + step) : (segment->contourB.size() - 1)];
			tangent = forward - backward;

			result.push_back(Point(-1, FindPerpPoint(&(segment->center), segment->contourB[index], tangent, perp, a, b)));
		}


		a = result[i].y + length * 0.3;
	}

	return 1;
}

int FindPerpPoint(vector<Point>* input, Point x, Point targent, Point &result, int startindex, int endindex)
{
	if (abs((*input)[0].x - x.x) + abs((*input)[0].y - x.y) <1)
	{
		result = x;
		return 0;
	}
	else if (abs((*input)[input->size() - 1].x - x.x) + abs((*input)[input->size() - 1].y - x.y) <1)
	{
		result = x;
		return input->size() - 1;
	}

	int mindot = INT_MAX;
	int temp, index = startindex;
	Point pt;

	startindex = startindex >= 0 ? startindex : 0;
	startindex = startindex <= input->size() - 1 ? startindex : input->size() - 1;
	endindex = endindex <= input->size() - 1 ? endindex : input->size() - 1;


	for (int i = startindex; i <= endindex; i++)
	{
		pt = (*input)[i] - x;
		temp = pt.x * targent.x + pt.y * targent.y;
		temp = abs(temp);
		if (mindot > temp)
		{
			mindot = temp;
			index = i;
		}
	}

	result = (*input)[index];
	return index;
}

int FindMax(vector<double>* input, int start, int end)
{
	double Max = -100;
	int indexMax = start;

	for (int i = start; i < end; i++)
	{
		if ((*input)[i] > Max)
		{
			indexMax = i;
			Max = (*input)[i];
		}
	}
	return indexMax;
}

int FindMin(vector<double>* input, int start, int end)
{
	double Min = 100;
	int indexMin = start;

	for (int i = start; i < end; i++)
	{
		if ((*input)[i] < Min)
		{
			indexMin = i;
			Min = (*input)[i];
		}
	}
	return indexMin;
}

int CurvatureAnalysis(vector<double>* curvature, vector<Point>& cur)
{
	if (curvature->size() == 0)
	{
		return 0;
	}


	vector<double> curv, scurv, super_scurv;
	curv.insert(curv.begin(), curvature->begin() + 1, curvature->end() - 1);

	Smooth1DSequence(&curv, scurv, 1);   
	Smooth1DSequence(&curv, super_scurv, 3); 

	float omit =0.08;
	for (int i = 0; i < curv.size()*omit; i++)
	{
		scurv[i] = 0;
		super_scurv[i] = 0;
	}
	for (int i = curv.size()*(1-omit); i < curv.size(); i++)
	{
		scurv[i] = 0;
		super_scurv[i] = 0;
	}


	vector<int> pre_Zeros, Zeros;
	vector<int>	pre_Max, pre_Min;
	Point Headpeak = Point(0, 0);
	Point Tailpeak = Point(0, 0);


	for (int i = 0; i < scurv.size() - 1; i++)
	{
		if (scurv[i] == 0 || scurv[i + 1] * scurv[i] <0) 
		{
			pre_Zeros.push_back(i);

		}

	}


	for (int i = 0; i < pre_Zeros.size() - 1; i++)
	{
		if (pre_Zeros[i + 1] - pre_Zeros[i] > curv.size() / 6 && scurv[pre_Zeros[i] + 1] > 0) 
		{
			double Max = 0;
			int indexMax = pre_Zeros[i] + 1;
			for (int j = pre_Zeros[i] + 1; j < pre_Zeros[i + 1] - 1; j++)
			{
				if (Max < super_scurv[j])  
				{
					Max = super_scurv[j];
					indexMax = j;
				}
			}
			pre_Max.push_back(indexMax);

		
			if (Zeros.size() == 0)
			{
				Zeros.push_back(pre_Zeros[i]);
				Zeros.push_back(pre_Zeros[i + 1]);
			}
			else
			{
				if (Zeros[Zeros.size() - 1] != pre_Zeros[i])
				{
					Zeros.push_back(pre_Zeros[i]);
					Zeros.push_back(pre_Zeros[i + 1]);
				}
				else
				{
					Zeros.push_back(pre_Zeros[i + 1]);
				}
			}

		}

		if (pre_Zeros[i + 1] - pre_Zeros[i] > curv.size() / 6 && scurv[pre_Zeros[i] + 1] < 0) 
		{
			double Min = 100;
			int indexMin = pre_Zeros[i] + 1;
			for (int j = pre_Zeros[i] + 1; j < pre_Zeros[i + 1] - 1; j++)
			{
				if (Min > super_scurv[j]) 
				{
					Min = super_scurv[j];
					indexMin = j;
				}
			}
			pre_Min.push_back(indexMin);
		
			if (Zeros.size() == 0)
			{
				Zeros.push_back(pre_Zeros[i]);
				Zeros.push_back(pre_Zeros[i + 1]);
			}
			else
			{
				if (Zeros[Zeros.size() - 1] != pre_Zeros[i])
				{
					Zeros.push_back(pre_Zeros[i]);
					Zeros.push_back(pre_Zeros[i + 1]);
				}
				else
				{
					Zeros.push_back(pre_Zeros[i + 1]);
				}
			}

		}
	}




	if (Zeros.size() != 0)
	{
		if (Zeros[0]>0)
		{
			if (scurv[Zeros[0] - 1] >0)
			{
				double Max = 0;
				int indexMax = 1;
				for (int j = 1; j < Zeros[0] - 1; j++)
				{
					if (Max < super_scurv[j])  
					{
						Max = super_scurv[j];
						indexMax = j;
					}
				}
				Headpeak = Point(1, indexMax);
			}
			else
			{
				double Min = 100;
				int indexMin = 0;
				for (int j = 1; j < Zeros[0] - 1; j++)
				{
					if (Min > super_scurv[j])  
					{
						Min = super_scurv[j];
						indexMin = j;
					}
				}
				Headpeak = Point(-1, indexMin);
			}
		}
	}


	if (Zeros.size() != 0)
	{
		if (Zeros[Zeros.size() - 1] < scurv.size() - 1)
		{
			if (scurv[Zeros[Zeros.size() - 1] + 1] >0)
			{
				double Max = 0;
				int indexMax = Zeros[Zeros.size() - 1] + 1;
				for (int j = Zeros[Zeros.size() - 1] + 1; j < scurv.size() - 2; j++)
				{
					if (Max < super_scurv[j])  
					{
						Max = super_scurv[j];
						indexMax = j;
					}
				}
				Tailpeak = Point(1, indexMax);
			}
			else
			{
				double Min = 100;
				int indexMin = Zeros[Zeros.size() - 1] + 1;
				for (int j = Zeros[Zeros.size() - 1] + 1; j < scurv.size() - 2; j++)
				{
					if (Min > super_scurv[j])
					{
						Min = super_scurv[j];
						indexMin = j;
					}
				}
				Tailpeak = Point(-1, indexMin);
			}

		}
	}


	if (Headpeak != Point(0, 0) && Zeros.size() != 0)
	{
		if (Zeros[0] - Headpeak.y > scurv.size() / 8)
		{
			if (Headpeak.x > 0)
			{
				pre_Max.insert(pre_Max.begin(), Headpeak.y);
			}
			else
			{
				pre_Min.insert(pre_Min.begin(), Headpeak.y);
			}
		}
	}

	if (Tailpeak != Point(0, 0) && Zeros.size() != 0)
	{
		if (Tailpeak.y - Zeros[Zeros.size() - 1] > scurv.size() / 8)
		{
			if (Tailpeak.x > 0)
			{
				pre_Max.push_back(Tailpeak.y);
			}
			else
			{
				pre_Min.push_back(Tailpeak.y);
			}
		}
	}

	for (int i = 0; i < pre_Max.size(); i++)
	{
		cur.push_back(Point(1, pre_Max[i] + 1));
	}
	for (int i = 0; i < pre_Min.size(); i++)
	{
		cur.push_back(Point(-1, pre_Min[i] + 1));
	}
	for (int i = 0; i < Zeros.size(); i++)
	{
		cur.push_back(Point(0, Zeros[i] + 1));
	}

	if (cur.size() != 0)
	{
		for (int i = 0; i < cur.size() - 1; i++)
		{ 
			for (int j = 0; j < cur.size() - i - 1; j++)
			{ 
				if (cur[j].y > cur[j + 1].y)
				{
					Point temp = cur[j];
					cur[j] = cur[j + 1];
					cur[j + 1] = temp;
				}
			}
		}
	}

	return 1;
}

void Smooth1DSequence(const vector<double>* input, vector<double>& output, double sigma)
{
	int *kernel, klength, normfactor;

	int ll, ul, x;
	double n;
	ll = (int)(-3 * sigma) - 1;
	ul = (int)(3 * sigma) + 1;
	klength = ul - ll + 1;
	kernel = (int*)malloc(klength * sizeof(int));

	normfactor = 0;
	n = exp(-1.0*ll*ll / (2 * sigma*sigma));
	for (x = 0; x < klength; x++)
	{
		(kernel)[x] = (int)(exp(-1.0*(x + ll)*(x + ll) / (2 * sigma*sigma)) / n + 0.5);
		normfactor += (kernel)[x];
	}


	int j, k, ind, anchor;
	double sum;

	double *pt;
	int length = input->size();
	pt = (double*)malloc(length * sizeof(Point));
	anchor = klength / 2;
	for (j = 0; j < length; j++)
	{
		sum = 0;
		for (k = 0; k < klength; k++)
		{
			ind = j + k - anchor;
			ind = ind > 0 ? ind : 0;
			ind = ind < length ? ind : (length - 1);
			sum = sum + (*input)[ind] * kernel[k];
		}

		pt[j] = (double)(1.0*sum / normfactor);

	}
	output = vector<double>(pt, pt + length);

	free(pt);

}

int DrawImage(seg& segment, Mat& I, int partnum, vector<Point> cur)
{
	
	float offsite = 0.2;
	float extension = 1;

	if (segment.center.size() == 0)
	{
		return 0;
	}

	vector<Point> newCenter, newContourA, newContourB;
	for (int i = 0; i < segment.center.size(); i++)
	{
		newCenter.push_back(segment.center[i] - m_A);
	}

	for (int i = 0; i < segment.contourA.size(); i++)
	{
		newContourA.push_back(segment.contourA[i] - m_A);
	}

	for (int i = 0; i < segment.contourB.size(); i++)
	{
		newContourB.push_back(segment.contourB[i] - m_A);
	}

	Mat roi = I(cv::Rect(m_A, m_B));
	Mat overlay;
	roi.copyTo(overlay);

	vector<Point> controlA, controlB;
	for (int i = 0; i < partnum; i++)
	{
		if (i % 3 == 1)
		{
			controlA.push_back(Point(i, 0));
			controlB.push_back(Point(i, 0));
		}
		else
		{
			controlA.push_back(Point(i, 255));
			controlB.push_back(Point(i, 255));
		}
	}

	vector<Point> SegContours;

	Point Int, Ext;
	vector<Point> SegCountours, BoundA, BoundB;
	for (int i = 0; i < segment.Partnum; i++)     
	{
		if (controlA[i].y != 0)
		{
			do{
				Int = offsite*(newContourA[segment.segAB[i].x] - newCenter[i]) + newCenter[i];
				SegCountours.push_back(Int);

				Ext = extension*(newContourA[segment.segAB[i].x] - newCenter[i]) + newCenter[i];
				BoundA.push_back(Ext);

				i++;
				if (i == segment.Partnum)
					break;
			} while (controlA[i].y == controlA[i - 1].y);

			Int = offsite*(newContourA[segment.segAB[i].x] - newCenter[i]) + newCenter[i];
			SegCountours.push_back(Int);   

			Ext = extension*(newContourA[segment.segAB[i].x] - newCenter[i]) + newCenter[i];
			BoundA.push_back(Ext);
			SegCountours.insert(SegCountours.end(), BoundA.rbegin(), BoundA.rend());


			SegCountours.clear();
			BoundA.clear();
		}
	}

	for (int i = 0; i < segment.Partnum; i++)
	{
		if (controlB[i].y != 0)
		{
			do{
				Int = offsite*(newContourB[segment.segAB[i].y] - newCenter[i]) + newCenter[i];
				SegCountours.push_back(Int);

				Ext = extension*(newContourB[segment.segAB[i].y] - newCenter[i]) + newCenter[i];

				BoundB.push_back(Ext);
				i++;
				if (i == segment.Partnum)
					break;
			} while (controlB[i].y == controlB[i - 1].y);

			Int = offsite*(newContourB[segment.segAB[i].y] - newCenter[i]) + newCenter[i];
			SegCountours.push_back(Int);    

			Ext = extension*(newContourB[segment.segAB[i].y] - newCenter[i]) + newCenter[i];

			BoundB.push_back(Ext);
			SegCountours.insert(SegCountours.end(), BoundB.rbegin(), BoundB.rend());



			SegCountours.clear();
			BoundB.clear();
		}
	}


	for (int i = 0; i < segment.contourA.size() - 1; i++)
	{
	
	}
	for (int i = 0; i < segment.contourB.size() - 1; i++)
	{
	
	}



	circle(overlay, newCenter[12], 5, Scalar(255, 0, 255), -1, 8, 0);

	circle(overlay, segment.head - m_A, 5, Scalar(0, 0, 255), -1, 8, 0);


	circle(overlay, segment.tail - m_A, 5, Scalar(0, 255, 0), -1, 8, 0);


	for (int i = 0; i < cur.size(); i++)
	{
		if (cur[i].x == 0)
		{
			
			circle(overlay, segment.center[cur[i].y] - m_A, 5, Scalar(255, 255, 0), -1, 8, 0);
		}
		else if (cur[i].x == -1)
		{
		
			circle(overlay, segment.center[cur[i].y] - m_A, 5, Scalar(0, 255, 255), -1, 8, 0);
		}
		else
		{
		
			circle(overlay, segment.center[cur[i].y] - m_A, 5, Scalar(0, 255, 255), -1, 8, 0);
		}
	}

	float alpha = 0.5;
	addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi);

	return 1;
}

