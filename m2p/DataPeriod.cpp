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