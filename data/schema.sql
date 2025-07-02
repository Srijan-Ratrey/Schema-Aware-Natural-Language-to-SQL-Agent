-- Sample Database Schema for NL2SQL Project
-- This schema creates tables for a typical e-commerce system

-- Customer table
CREATE TABLE IF NOT EXISTS Customer (
    Id INTEGER PRIMARY KEY,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    City VARCHAR(50),
    Country VARCHAR(50),
    Phone VARCHAR(20)
);

-- Supplier table
CREATE TABLE IF NOT EXISTS Supplier (
    Id INTEGER PRIMARY KEY,
    CompanyName VARCHAR(100) NOT NULL,
    ContactName VARCHAR(100),
    City VARCHAR(50),
    Country VARCHAR(50),
    Phone VARCHAR(20),
    Fax VARCHAR(20)
);

-- Product table
CREATE TABLE IF NOT EXISTS Product (
    Id INTEGER PRIMARY KEY,
    ProductName VARCHAR(100) NOT NULL,
    SupplierId INTEGER,
    UnitPrice DECIMAL(10,2),
    Package VARCHAR(100),
    IsDiscontinued BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (SupplierId) REFERENCES Supplier(Id)
);

-- Order table
CREATE TABLE IF NOT EXISTS [Order] (
    Id INTEGER PRIMARY KEY,
    OrderDate DATE,
    OrderNumber VARCHAR(20),
    CustomerId INTEGER,
    TotalAmount DECIMAL(10,2),
    FOREIGN KEY (CustomerId) REFERENCES Customer(Id)
);

-- OrderItem table (junction table for Order-Product relationship)
CREATE TABLE IF NOT EXISTS OrderItem (
    Id INTEGER PRIMARY KEY,
    OrderId INTEGER,
    ProductId INTEGER,
    UnitPrice DECIMAL(10,2),
    Quantity INTEGER,
    FOREIGN KEY (OrderId) REFERENCES [Order](Id),
    FOREIGN KEY (ProductId) REFERENCES Product(Id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_country ON Customer(Country);
CREATE INDEX IF NOT EXISTS idx_customer_city ON Customer(City);
CREATE INDEX IF NOT EXISTS idx_product_supplier ON Product(SupplierId);
CREATE INDEX IF NOT EXISTS idx_order_customer ON [Order](CustomerId);
CREATE INDEX IF NOT EXISTS idx_orderitem_order ON OrderItem(OrderId);
CREATE INDEX IF NOT EXISTS idx_orderitem_product ON OrderItem(ProductId); 