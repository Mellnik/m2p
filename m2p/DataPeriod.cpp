#include "main.h"
#include "DataPeriod.h"

DataPeriod::DataPeriod(size_t Size)
{
	m_Size = Size;
	m_Points.reserve(Size);
}

DataPeriod::~DataPeriod()
{

}

void DataPeriod::AddDataPoint(DataPoint *point)
{
	m_Points.push_back(point);
}

bool DataPeriod::IsFull()
{
	return (m_Points.size() == m_Size);
}

void DataPeriod::Debug()
{
	int PointCount = 0;
	for (auto Point : m_Points)
	{
		cout << " Point " << ++PointCount << ":" << endl;
		cout.precision(std::numeric_limits<double>::max_digits10);
		double delta = Point->m_Close - Point->m_Open;
		cout << "    Date " << Point->m_Date << ", Delta " << std::fixed << delta << endl;
		//cout << "    Date " << Point->m_Date << ", Open " << std::fixed << Point->m_Open << ", Close " << std::fixed << Point->m_Close << endl;
	}
}