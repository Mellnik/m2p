#ifndef __DATAPOINT_H
#define __DATAPOINT_H

class DataPoint
{
public:
	DataPoint();
	~DataPoint();
	
	int m_Date;
	float m_High;
	float m_Low;
	float m_Open;
	float m_Close;
	float m_Volume;
	float m_CoinVolume;
	float m_WeightedAverage;

private:
};

#endif /* __DATAPOINT_H */